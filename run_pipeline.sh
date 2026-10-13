#!/bin/bash

# Definição de cores para organizar o *output* no terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sem Cor

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE} PIPELINE DE SIMULAÇÃO 1PN-EIH - PREPARAÇÃO DO MANUSCRITO Q1    ${NC}"
echo -e "${BLUE}================================================================${NC}"

# ==============================================================================
# FASE 1: Dinâmica Temporal e Ondas Gravitacionais
# ==============================================================================
echo -e "\n${YELLOW}[FASE 1/3] Iniciando Dinâmica Temporal e Ondas Gravitacionais...${NC}"
echo "-> A executar simulador Julia (gerar_dados_ondas_gravitacionais.jl)..."
julia gerar_dados_ondas_gravitacionais.jl

echo "-> A gerar gráficos Python (plotar_painel_astrofisico.py)..."
# Certifique-se de que este ficheiro contém o código Python real e não o Julia!
python3 plotar_painel_astrofisico.py


# ==============================================================================
# FASE 2: Topologia Fractal e Bacias de Atração
# ==============================================================================
echo -e "\n${YELLOW}[FASE 2/3] Iniciando Mapeamento Topológico de Bacias HD...${NC}"
echo "-> A executar simulador Julia em GPU (gerar_matriz_bacias.jl)..."
julia gerar_matriz_bacias.jl

echo "-> A renderizar mapa fractal Dark Academic Python (plotar_bacias_wada.py)..."
python3 plotar_bacias_wada.py


# ==============================================================================
# FASE 3: Quantificação Estatística (Incerteza e Wada)
# ==============================================================================
echo -e "\n${YELLOW}[FASE 3/3] Iniciando Varreduras Estatísticas e Parâmetros Críticos...${NC}"

echo "-> A executar varredura da Excentricidade (sweep_excentricidade.jl)..."
# Este script gera o seu próprio gráfico PNG através do CairoMakie
julia sweep_excentricidade.jl

echo "-> A executar varreduras paramétricas completas (sweep_parametros_criticos.jl)..."
julia sweep_parametros_criticos.jl

echo "-> A corrigir os nomes dos ficheiros importados no script Python..."
# Corrige automaticamente o erro no seu ficheiro Python original
sed -i 's/dados_sweep_CPN_e048.txt/metricas_completas_CPN_e048.txt/g' plot_critical_parameters.py
sed -i 's/dados_sweep_mu_e048.txt/metricas_completas_mu_e048.txt/g' plot_critical_parameters.py
sed -i 's/dados_sweep_c_e048.txt/metricas_completas_c_e048.txt/g' plot_critical_parameters.py

echo "-> A compilar o painel 3x3 de métricas estatísticas (plot_critical_parameters.py)..."
python3 plot_critical_parameters.py


echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN} PIPELINE CONCLUÍDO COM SUCESSO! ${NC}"
echo -e "${GREEN} Todos os dados e figuras Q1 foram gerados e estão na pasta.${NC}"
echo -e "${GREEN}================================================================${NC}"
