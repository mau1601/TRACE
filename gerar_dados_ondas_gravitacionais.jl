# ==============================================================================
# EXTRAÇÃO DE ONDAS GRAVITACIONAIS (h_+(t)) - SISTEMA 1PN-EIH
# Versão com Sincronia Temporal Completa entre Regular e Caótico (e = 0.48)
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
    r_esc::Float32   = 30.0f0          # Limiar de escape ligeiramente alargado
    r_coll::Float32  = 0.012f0         # Tolerância de colisão refinada
    dt::Float32      = 0.0016f0        # Passo temporal otimizado para alta resolução
    max_steps::Int32 = 2200            # Assegura a cobertura temporal total (t ~ 3.5+)
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

function simular_trajetoria_completa(x0, y0, p::SimParams)
    v_apo = sqrt((1f0 - p.e) / (1f0 + p.e))
    r1 = SVector{3,Float32}(-p.m2, 0f0, 0f0); v1 = SVector{3,Float32}(0f0, -p.m2 * v_apo, 0f0)
    r2 = SVector{3,Float32}(p.m1, 0f0, 0f0);  v2 = SVector{3,Float32}(0f0, p.m1 * v_apo, 0f0)
    r3 = SVector{3,Float32}(x0, y0, 0f0);     v3 = SVector{3,Float32}(0f0, 0f0, 0f0)
    
    t_hist = Float32[]
    h_plus = Float32[]
    dist_R = 100.0f0 
    
    for step in 1:p.max_steps
        a1, a2, a3 = calc_accel_1pn(r1, r2, r3, v1, v2, v3, p.m1, p.m2, p.m3, p.G, p.c)
        
        # Fórmula Quadrupolar 1PN (Strain h_+)
        d2I_xx = p.m1*(2f0*v1[1]^2 + 2f0*r1[1]*a1[1]) + p.m2*(2f0*v2[1]^2 + 2f0*r2[1]*a2[1]) + p.m3*(2f0*v3[1]^2 + 2f0*r3[1]*a3[1])
        d2I_yy = p.m1*(2f0*v1[2]^2 + 2f0*r1[2]*a1[2]) + p.m2*(2f0*v2[2]^2 + 2f0*r2[2]*a2[2]) + p.m3*(2f0*v3[2]^2 + 2f0*r3[2]*a3[2])
        
        hp = (p.G / (p.c^4 * dist_R)) * (d2I_xx - d2I_yy)
        
        current_t = step * p.dt
        push!(t_hist, current_t)
        push!(h_plus, hp)
        
        v1 += a1 * p.dt; r1 += v1 * p.dt
        v2 += a2 * p.dt; r2 += v2 * p.dt
        v3 += a3 * p.dt; r3 += v3 * p.dt
        
        if gpu_norm(r3 - r1) < p.r_coll || gpu_norm(r3 - r2) < p.r_coll || gpu_norm(r3) > p.r_esc
            break
        end
    end
    return t_hist, h_plus
end

function main()
    p = SimParams()
    
    println(">>> A simular órbita Regular...")
    t_reg, h_reg = simular_trajetoria_completa(1.6f0, 1.2f0, p)
    
    println(">>> A simular órbita Caótica de Longa Duração Sincronizada...")
    # Coordenada ajustada para transiente caótico prolongado
    t_ch, h_ch = simular_trajetoria_completa(-0.05f0, 0.85f0, p)
    
    writedlm("onda_regular.txt", hcat(t_reg, h_reg), ',')
    writedlm("onda_caotica.txt", hcat(t_ch, h_ch), ',')
    println(">>> Ficheiros atualizados com sincronia temporal!")
end

main()
