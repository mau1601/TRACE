# ==============================================================================
# VARREDURAS PARAMÉTRICAS 1PN NO PICO DE CAOS (e = 0.48) - MÉTRICAS COMPLETAS
# Análise de C_PN, Assimetria (μ) e Parâmetro Relativístico (c)
# Métricas: Expoente α, Entropia Sb, Wada (W12, W13, W23), Fração de Escape
# Implementação: CUDA.jl + CairoMakie
# ==============================================================================

using CUDA
using StaticArrays
using CairoMakie
using Statistics
using DelimitedFiles
using Printf

# ------------------------------------------------------------------------------
# 1. PARÂMETROS BASE DA SIMULAÇÃO
# ------------------------------------------------------------------------------
Base.@kwdef struct SimParams
    G::Float32       = 1.0f0
    c::Float32       = 20.0f0
    m1::Float32      = 0.9f0
    m2::Float32      = 0.1f0
    m3::Float32      = 1f-5
    
    e::Float32       = 0.48f0          # FIXO NO PICO DE CAOS ABSOLUTO
    C_PN::Float32    = 2.80f0          # Pseudo-constante de Jacobi
    
    r_esc::Float32   = 25.0f0
    r_coll::Float32  = 0.02f0          # Tolerância fina para alta excentricidade
    
    dt::Float32      = 0.005f0
    max_steps::Int32 = 8000            # Suficiente para capturar a topologia assintótica
    
    N::Int32         = 1024            # Resolução balanceada para múltiplas varreduras
    grid_min::Float32 = -2.0f0
    grid_max::Float32 = 2.0f0
end

@inline gpu_dot(a::SVector{3, Float32}, b::SVector{3, Float32}) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline gpu_norm(a::SVector{3, Float32}) = sqrt(a[1]^2 + a[2]^2 + a[3]^2)

# ------------------------------------------------------------------------------
# 2. EQUAÇÕES 1PN (GPU)
# ------------------------------------------------------------------------------
@inline function get_accel(target_idx::Int32, pos, vel, masses, G::Float32, c2::Float32)
    a0 = SVector{3, Float32}(0f0, 0f0, 0f0)
    a1_pn = SVector{3, Float32}(0f0, 0f0, 0f0)
    va = vel[target_idx]
    ma = masses[target_idx]
    
    for b in Int32(1):Int32(3)
        if target_idx != b
            vb = vel[b]
            mb = masses[b]
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
    pos = (r1, r2, r3)
    vel = (v1, v2, v3)
    masses = (m1, m2, m3)
    c2 = c^2
    return get_accel(Int32(1), pos, vel, masses, G, c2), 
           get_accel(Int32(2), pos, vel, masses, G, c2), 
           get_accel(Int32(3), pos, vel, masses, G, c2)
end

# ------------------------------------------------------------------------------
# 3. KERNEL DE VARREDURA
# ------------------------------------------------------------------------------
function kernel_sweep_bacias!(basin, x_grid, y_grid, p::SimParams)
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
            basin[idx, idy] = Int8(-1)
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
                outcome = Int8(1); break
            elseif gpu_norm(r3 - r2) < p.r_coll
                outcome = Int8(2); break
            elseif gpu_norm(r3) > p.r_esc
                outcome = Int8(3); break
            end
        end
        basin[idx, idy] = outcome
    end
    return nothing
end

# ------------------------------------------------------------------------------
# 4. CÁLCULO COMPLETO DE TODAS AS MÉTRICAS (α, Sb, Wada, Escape)
# ------------------------------------------------------------------------------
function calc_all_metrics(basin_cpu, N)
    # A. Coeficiente de Incerteza (α)
    incertos = 0
    valid_pixels = 0
    for i in 1:(N-1), j in 1:(N-1)
        if basin_cpu[i,j] != -1 && basin_cpu[i+1,j] != -1 && basin_cpu[i,j+1] != -1
            valid_pixels += 1
            if basin_cpu[i,j] != basin_cpu[i+1,j] || basin_cpu[i,j] != basin_cpu[i,j+1]
                incertos += 1
            end
        end
    end
    alpha = valid_pixels > 0 ? log((incertos / valid_pixels) + 1e-10) / log(1.0 / N) : 0.0
    alpha = clamp(alpha, 0.0, 1.0)

    # B. Entropia da Bacia (Sb)
    entropies = Float64[]
    step_sz = 8
    for i in 1:step_sz:(N-step_sz), j in 1:step_sz:(N-step_sz)
        sub = basin_cpu[i:i+step_sz-1, j:j+step_sz-1]
        # Considera atratores 0 (KAM/Proibido), 1, 2, 3
        total_sub = length(sub)
        cts = [count(==(c), sub) / total_sub for c in 0:3]
        push!(entropies, -sum(px > 0 ? px * log(px) : 0.0 for px in cts))
    end
    Sb = length(entropies) > 0 ? mean(entropies) : 0.0

    # C. Topologia de Wada (W12, W13, W23) Corrigida
    b12_from_1 = 0; b13_from_1 = 0; b1_total = 0
    b12_from_2 = 0; b23_from_2 = 0; b2_total = 0
    
    for i in 2:(N-1), j in 2:(N-1)
        c = basin_cpu[i,j]
        if c <= 0; continue; end
        vizinhos = (basin_cpu[i-1,j], basin_cpu[i+1,j], basin_cpu[i,j-1], basin_cpu[i,j+1])
        if c == 1
            has_2 = 2 in vizinhos; has_3 = 3 in vizinhos
            if has_2; b12_from_1 += 1; end
            if has_3; b13_from_1 += 1; end
            if has_2 || has_3; b1_total += 1; end
        elseif c == 2
            has_1 = 1 in vizinhos; has_3 = 3 in vizinhos
            if has_1; b12_from_2 += 1; end
            if has_3; b23_from_2 += 1; end
            if has_1 || has_3; b2_total += 1; end
        end
    end
    
    W_12 = b1_total > 0 ? (b12_from_1 / b1_total) : 0.0
    W_13 = b1_total > 0 ? (b13_from_1 / b1_total) : 0.0
    W_23 = b2_total > 0 ? (b23_from_2 / b2_total) : 0.0

    # D. Fração de Escape
    total_valid = max(1, count(!=(-1), basin_cpu))
    f_esc = count(==(3), basin_cpu) / total_valid

    return alpha, Sb, W_12, W_13, W_23, f_esc
end

# ------------------------------------------------------------------------------
# 5. ROTINAS DE VARREDURA COMPLETA
# ------------------------------------------------------------------------------
function main()
    println(">>> ALOCANDO RECURSOS GPU (e = 0.48 Fixo - Métricas Completas)...")
    p_base = SimParams(e=0.48f0)
    x_coords = Float32.(range(p_base.grid_min, stop=p_base.grid_max, length=p_base.N))
    y_coords = Float32.(range(p_base.grid_min, stop=p_base.grid_max, length=p_base.N))
    d_x = CuArray(x_coords); d_y = CuArray(y_coords)
    d_basin = CUDA.zeros(Int8, p_base.N, p_base.N)
    threads = (16, 16); blocks = (cld(p_base.N, 16), cld(p_base.N, 16))

    # --- VARREDURA 1: C_PN ---
    println("\n>>> Varredura 1: C_PN [2.50 a 4.20]")
    cpn_vals = Float32.(collect(2.50:0.02:4.20))
    res_cpn = Float64[]
    for v in cpn_vals
        @cuda threads=threads blocks=blocks kernel_sweep_bacias!(d_basin, d_x, d_y, SimParams(e=0.48f0, C_PN=v))
        CUDA.synchronize()
        basin = Array(d_basin)
        alpha, Sb, w12, w13, w23, f_esc = calc_all_metrics(basin, p_base.N)
        push!(res_cpn, v)
        @printf("CPN=%.2f | α=%.4f | Sb=%.4f | W13=%.2f | Esc=%.2f%%\n", v, alpha, Sb, w13, f_esc*100)
        # Vamos acumular linha temporária para exportar
    end
    # Nota: No script real, guardamos cada métrica em vetores separados para o writedlm.
    
    # Abaixo estruturamos de forma limpa para exportar todas as colunas: [Param, Alpha, Sb, W12, W13, W23, Escape]
    function run_full_sweep(param_name, values, param_gen_func)
        out_data = Float64[]
        for v in values
            p = param_gen_func(v)
            @cuda threads=threads blocks=blocks kernel_sweep_bacias!(d_basin, d_x, d_y, p)
            CUDA.synchronize()
            basin = Array(d_basin)
            alpha, Sb, w12, w13, w23, f_esc = calc_all_metrics(basin, p_base.N)
            append!(out_data, [v, alpha, Sb, w12, w13, w23, f_esc])
        end
        return reshape(out_data, 7, length(values))'
    end

    println("\n>>> Executando Varredura C_PN...")
    data_cpn = run_full_sweep("C_PN", cpn_vals, v -> SimParams(e=0.48f0, C_PN=v))
    writedlm("metricas_completas_CPN_e048.txt", data_cpn, ',')

    println("\n>>> Executando Varredura μ...")
    mu_vals = Float32.(collect(0.05:0.01:0.50))
    data_mu = run_full_sweep("mu", mu_vals, v -> SimParams(e=0.48f0, m1=1.0f0-v, m2=v))
    writedlm("metricas_completas_mu_e048.txt", data_mu, ',')

    println("\n>>> Executando Varredura c...")
    c_vals = Float32.(collect(10.0:2.0:100.0))
    data_c = run_full_sweep("c", c_vals, v -> SimParams(e=0.48f0, c=v))
    writedlm("metricas_completas_c_e048.txt", data_c, ',')

    println("\n>>> Todas as métricas calculadas e exportadas com sucesso!")
end

main()
