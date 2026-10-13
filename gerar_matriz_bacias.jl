# ==============================================================================
# MOPAGEM DE BACIAS DE ATRAÇÃO - ZOOM FRACTAL DE ALTA DENSIDADE (N = 3000)
# ==============================================================================

using CUDA
using StaticArrays
using DelimitedFiles
using Printf

Base.@kwdef struct SimParams
    G::Float32       = 1.0f0
    c::Float32       = 20.0f0
    m1::Float32      = 0.9f0
    m2::Float32      = 0.1f0
    m3::Float32      = 1f-5
    e::Float32       = 0.48f0          # Pico de Caos Crítico
    C_PN::Float32    = 2.80f0          
    r_esc::Float32   = 30.0f0          
    r_coll::Float32  = 0.012f0         
    dt::Float32      = 0.003f0         
    max_steps::Int32 = 4500            
    N::Int32         = 10000            # Alta densidade de pontos (3000x3000)
    grid_min::Float32 = -2.4f0         # Zoom focado na região central fractal
    grid_max::Float32 =  2.4f0         
    grid_ymin::Float32 = -2.4f0        
    grid_ymax::Float32 =  2.4f0        
end

@inline gpu_dot(a::SVector{3, Float32}, b::SVector{3, Float32}) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline gpu_norm(a::SVector{3, Float32}) = sqrt(a[1]^2 + a[2]^2 + a[3]^2)

@inline function get_accel(target_idx::Int32, pos, vel, masses, G::Float32, c2::Float32)
    a0 = SVector{3, Float32}(0f0, 0f0, 0f0)
    a1_pn = SVector{3, Float32}(0f0, 0f0, 0f0)
    va = vel[target_idx]; ma = masses[target_idx]
    
    for b in Int32(1):Int32(3)
        if target_idx != b
            vb = vel[b]; mb = masses[b]
            rab_vec = pos[target_idx] - pos[b]
            rab = gpu_norm(rab_vec) + 1f-6
            nab = rab_vec / rab
            vab = va - vb
            a0 += -(G * mb / rab^2) * nab
            
            term1 = -gpu_dot(va, va) - 2f0*gpu_dot(vb, vb) + 4f0*gpu_dot(va, vb) + 1.5f0*(gpu_dot(nab, vb))^2
            term1 += 5f0 * G * ma / rab + 4f0 * G * mb / rab
            
            term2 = 0f0
            for c_idx in Int32(1):Int32(3)
                if c_idx != target_idx && c_idx != b
                    mc = masses[c_idx]
                    rac = gpu_norm(pos[target_idx] - pos[c_idx]) + 1f-6
                    rbc = gpu_norm(pos[b] - pos[c_idx]) + 1f-6
                    nbc = (pos[b] - pos[c_idx]) / rbc
                    term2 += G * mc * (4f0/rac + 1f0/rbc - (rab/(2f0*rbc^2))*gpu_dot(nab, nbc))
                end
            end
            a1_pn += (G * mb / rab^2) * (nab * (term1 + term2) + vab * gpu_dot(nab, 4f0*va - 3f0*vb))
        end
    end
    return a0 + a1_pn / c2
end

@inline function calc_accel_1pn(r1, r2, r3, v1, v2, v3, m1, m2, m3, G, c)
    pos = (r1, r2, r3); vel = (v1, v2, v3); masses = (m1, m2, m3); c2 = c^2
    return get_accel(Int32(1), pos, vel, masses, G, c2), 
           get_accel(Int32(2), pos, vel, masses, G, c2), 
           get_accel(Int32(3), pos, vel, masses, G, c2)
end

function kernel_basin_map!(basin, x_grid, y_grid, p::SimParams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if idx <= p.N && idy <= p.N
        x0 = x_grid[idx]
        y0 = y_grid[idy]
        
        v_apo = sqrt((1f0 - p.e) / (1f0 + p.e))
        r1 = SVector{3,Float32}(-p.m2, 0f0, 0f0)
        v1 = SVector{3,Float32}(0f0, -p.m2 * v_apo, 0f0)
        
        r2 = SVector{3,Float32}(p.m1, 0f0, 0f0)
        v2 = SVector{3,Float32}(0f0, p.m1 * v_apo, 0f0)
        
        r3 = SVector{3,Float32}(x0, y0, 0f0)
        
        dist_m1 = gpu_norm(r3 - r1) + 1f-6
        dist_m2 = gpu_norm(r3 - r2) + 1f-6
        ΩN = 0.5f0 * (x0^2 + y0^2) + p.m1 / dist_m1 + p.m2 / dist_m2
        vel_arg = 2.0f0 * ΩN - p.C_PN
        
        if vel_arg < 0.0f0
            basin[idx, idy] = Int8(-1) # Proibido / KAM
            return nothing
        end
        
        v3 = SVector{3,Float32}(0f0, sqrt(vel_arg), 0f0)
        outcome = Int8(0)
        
        for step in 1:p.max_steps
            a1, a2, a3 = calc_accel_1pn(r1, r2, r3, v1, v2, v3, p.m1, p.m2, p.m3, p.G, p.c)
            v1 += a1 * p.dt; r1 += v1 * p.dt
            v2 += a2 * p.dt; r2 += v2 * p.dt
            v3 += a3 * p.dt; r3 += v3 * p.dt
            
            if gpu_norm(r3 - r1) < p.r_coll
                outcome = Int8(1); break # Colisão Massa 1
            elseif gpu_norm(r3 - r2) < p.r_coll
                outcome = Int8(2); break # Colisão Massa 2
            elseif gpu_norm(r3) > p.r_esc
                outcome = Int8(3); break # Escape
            end
        end
        basin[idx, idy] = outcome
    end
    return nothing
end

function main()
    p = SimParams()
    x_coords = Float32.(range(p.grid_min, stop=p.grid_max, length=p.N))
    y_coords = Float32.(range(p.grid_ymin, stop=p.grid_ymax, length=p.N))
    
    d_x = CuArray(x_coords); d_y = CuArray(y_coords)
    d_basin = CUDA.zeros(Int8, p.N, p.N)
    
    threads = (32, 16)
    blocks = (cld(p.N, 32), cld(p.N, 16))
    
    println(">>> A calcular bacias em Zoom HD (3000x3000) na janela [-1.4, 1.4]...")
    @cuda threads=threads blocks=blocks kernel_basin_map!(d_basin, d_x, d_y, p)
    CUDA.synchronize()
    
    basin_cpu = Array(d_basin)
    writedlm("atratores_bacia.txt", basin_cpu, ',')
    println(">>> Ficheiro 'atratores_bacia.txt' gerado com sucesso!")
end

main()
