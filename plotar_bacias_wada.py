import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Configurações de estilo de publicação acadêmica (Q1)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# 1. Carregar a matriz de bacias gerada no range [-2.4, 2.4]
try:
    # Substitua pelo nome exato do ficheiro gerado pelo Julia
    basin_matrix = np.loadtxt('atratores_bacia.txt', delimiter=',') 
except IOError:
    print("Error: Ficheiro não encontrado.")
    exit(1)

fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

# 2. Paleta "Dark Academic"
# 0: Forbidden (Cinza escuro)
# 1: Unresolved / Ilhas KAM (Preto Profundo)
# 2: Collision B1 (Azul Académico)
# 3: Collision B2 (Vermelho Tijolo)
# 4: Escape B3 (Branco/Cinza claro)
cmap_dark = ListedColormap(['#222222', '#000000', '#0072B2', '#D55E00', '#F0F0F0'])

# Ajuste dos índices da matriz
basin_mapped = basin_matrix + 1

# 3. Renderizar o mapa com o extent ajustado para [-2.4, 2.4]
im = ax.imshow(
    basin_mapped.T, 
    origin='lower', 
    extent=[-2.4, 2.4, -2.4, 2.4], 
    cmap=cmap_dark, 
    interpolation='nearest',
    vmin=0, vmax=4
)

ax.set_title(r'Fractal Basins of Attraction ($e = 0.48$, 1PN)', fontweight='bold')
ax.set_xlabel(r'Initial Position $x_0$')
ax.set_ylabel(r'Initial Position $y_0$')

# 4. Configurar a barra de cores
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.ax.set_yticklabels([
    'Forbidden', 
    'Unresolved / KAM', 
    'Collision $\mathcal{B}_1$', 
    'Collision $\mathcal{B}_2$', 
    'Escape $\mathcal{B}_3$'
])

# 5. Gravar os ficheiros
output_pdf = 'basins_attraction_2k4_dark.pdf'
output_png = 'basins_attraction_2k4_dark.png'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Painel gerado com sucesso no range [-2.4, 2.4] como '{output_png}'!")
plt.show()
