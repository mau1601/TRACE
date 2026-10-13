# ==============================================================================
# VARREDURA 2D DE ULTRA-ALTA RESOLUÇÃO: Coeficiente de Incerteza (α) e Wada
# Implementação Otimizada: CUDA.jl + CairoMakie
# ==============================================================================

using CUDA
using StaticArrays
using CairoMakie
using Statistics
using DelimitedFiles
using Printf

# ------------------------------------------------------------------------------
# 1. PARÂMETROS REFINADOS PARA ALTA PERFORMANCE E PRECISÃO
# ------------------------------------------------------------------------------
Base.@kwdef struct SimParams
    G::Float32       = 1.0f0
    c::Float32       = 20.0f0
    m1::Float32      = 0.9f0
    m2::Float32      = 0.1f0
    m3::Float32      = 1f-5
    
    e::Float32       = 0.0f0           
    
    r_esc::Float32   = 25.0f0
    r_coll::Float32  = 0.02f0          # Suporta periastro extremo sem colisões falsas
    
    dt::Float32      = 0.005f0
    max_steps::Int32 = 8000            # Otimizado para não prender a GPU desnecessariamente
    
    N::Int32         = 2000            # ULTRA-ALTA RESOLUÇÃO (2000x2000)
    grid_min::Float32 = -2.0f0
    grid_max::Float32 = 2.0f0
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

function kernel_sweep_bacias!(basin, x_grid, y_grid, p::SimParams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if idx <= p.N && idy <= p.N
        x0 = x_grid[idx]; y0 = y_grid[idy]
        
        v_apo = sqrt((1f0 - p.e) / (1f0 + p.e))
        r1 = SVector{3,Float32}(-p.m2, 0f0, 0f0); v1 = SVector{3,Float32}(0f0, -p.m2 * v_apo, 0f0)
        r2 = SVector{3,Float32}(p.m1, 0f0, 0f0);  v2 = SVector{3,Float32}(0f0, p.m1 * v_apo, 0f0)
        r3 = SVector{3,Float32}(x0, y0, 0f0);     v3 = SVector{3,Float32}(0f0, 0f0, 0f0)
        
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
# CÁLCULO DE INCERTEZA (α) E WADA (W) - CORRIGIDO
# ------------------------------------------------------------------------------
function calc_alpha_and_wada(basin_cpu, N)
    # 1. Coeficiente de Incerteza (α)
    incertos = 0
    for i in 1:(N-1), j in 1:(N-1)
        if basin_cpu[i,j] != basin_cpu[i+1,j] || basin_cpu[i,j] != basin_cpu[i,j+1]
            incertos += 1
        end
    end
    f_eps = incertos / (N * N)
    alpha = log(f_eps + 1e-10) / log(1.0 / N)
    
    # 2. Parâmetro de Wada Corrigido (Contagem unilateral)
    b12_from_1 = 0; b13_from_1 = 0; b1_total = 0
    b12_from_2 = 0; b23_from_2 = 0; b2_total = 0
    
    for i in 2:(N-1), j in 2:(N-1)
        c = basin_cpu[i,j]
        if c == 0; continue; end # Ignora KAM
        
        vizinhos = (basin_cpu[i-1,j], basin_cpu[i+1,j], basin_cpu[i,j-1], basin_cpu[i,j+1])
        
        if c == 1
            has_2 = 2 in vizinhos
            has_3 = 3 in vizinhos
            if has_2; b12_from_1 += 1; end
            if has_3; b13_from_1 += 1; end
            if has_2 || has_3; b1_total += 1; end
            
        elseif c == 2
            has_1 = 1 in vizinhos
            has_3 = 3 in vizinhos
            if has_1; b12_from_2 += 1; end
            if has_3; b23_from_2 += 1; end
            if has_1 || has_3; b2_total += 1; end
        end
    end
    
    # Normalização rigorosa das frações
    W_12 = b1_total > 0 ? (b12_from_1 / b1_total) : 0.0
    W_13 = b1_total > 0 ? (b13_from_1 / b1_total) : 0.0
    W_23 = b2_total > 0 ? (b23_from_2 / b2_total) : 0.0
    
    return clamp(alpha, 0.0, 1.0), W_12, W_13, W_23
end

function main()
    # 121 pontos para detetar ressonâncias finas
    ecc_values = Float32.(collect(0.0:0.005:0.6))
    
    alphas = Float64[]
    w12_vals = Float64[]
    w13_vals = Float64[]
    w23_vals = Float64[]
    
    println(">>> Iniciando varredura Incerteza/Wada (", length(ecc_values), " pontos)...")
    println(">>> ALERTA: Resolução de 2000x2000 selecionada. O processamento exigirá tempo.")
    
    p_base = SimParams(e=0.0f0)
    x_coords = Float32.(range(p_base.grid_min, stop=p_base.grid_max, length=p_base.N))
    y_coords = Float32.(range(p_base.grid_min, stop=p_base.grid_max, length=p_base.N))
    d_x = CuArray(x_coords); d_y = CuArray(y_coords)
    d_basin = CUDA.zeros(Int8, p_base.N, p_base.N)
    
    threads = (16, 16)
    blocks = (cld(p_base.N, 16), cld(p_base.N, 16))
    
    for (i, ecc) in enumerate(ecc_values)
        p = SimParams(e=ecc)
        @cuda threads=threads blocks=blocks kernel_sweep_bacias!(d_basin, d_x, d_y, p)
        CUDA.synchronize()
        
        basin_cpu = Array(d_basin)
        alpha, w12, w13, w23 = calc_alpha_and_wada(basin_cpu, p.N)
        
        push!(alphas, alpha)
        push!(w12_vals, w12)
        push!(w13_vals, w13)
        push!(w23_vals, w23)
        
        @printf("Feito [%d/%d]: e = %.3f | α = %.4f | W(M1,M2) = %.4f | W(M1,Esc) = %.4f\n", i, length(ecc_values), ecc, alpha, w12, w13)
    end
    
    # Exportação de dados
    data_out = hcat(ecc_values, alphas, w12_vals, w13_vals, w23_vals)
    writedlm("dados_varredura_wada_alpha_121pontos_altares.txt", data_out, ',')
    println(">>> Dados guardados em dados_varredura_wada_alpha_121pontos_altares.txt")
    
    # Geração dos Gráficos
    fig = Figure(size = (1200, 500))
    
    # Painel Esquerdo: Coeficiente de Incerteza
    ax1 = Axis(fig[1, 1], title="Coeficiente de Incerteza (α) vs. Excentricidade", 
               xlabel="Excentricidade inicial (e)", ylabel="Expoente α")
    lines!(ax1, ecc_values, alphas, color=:darkorange, linewidth=2.0)
    scatter!(ax1, ecc_values, alphas, color=:darkorange, markersize=5)
    
    # Painel Direito: Parâmetros de Wada
    ax2 = Axis(fig[1, 2], title="Topologia de Wada (W) vs. Excentricidade", 
               xlabel="Excentricidade inicial (e)", ylabel="Fração de Fronteira Partilhada (W)")
    lines!(ax2, ecc_values, w12_vals, color=:purple, linewidth=2.5, label="Colisão M1 + Colisão M2 (W₁₂)")
    lines!(ax2, ecc_values, w13_vals, color=:red, linewidth=2.5, linestyle=:dash, label="Colisão M1 + Escape (W₁₃)")
    lines!(ax2, ecc_values, w23_vals, color=:blue, linewidth=2.5, linestyle=:dot, label="Colisão M2 + Escape (W₂₃)")
    axislegend(ax2, position=:rc)
    
    save("curvas_wada_incerteza_121pontos_altares.png", fig)
    println(">>> Gráficos gerados em curvas_wada_incerteza_121pontos_altares.png")
end

main()
