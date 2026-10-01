import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Force the non-interactive Agg backend
import matplotlib.pyplot as plt

def save_error_heatmap(depth_gt, depth_pred, mask, output_path, max_dist:int=10., max_error_viz=1.):
    """
    Gera o mapa de calor do erro absoluto usando um teto fixo para o limite superior da cor.
    
    :param max_error_viz: Valor em metros que representará o 'Vermelho Máximo' no colorbar.
    """
    # Calcular o mapa de erro absoluto bruto
    error_map = np.abs(depth_gt - depth_pred)
    
    # Forçar o erro a ser nan fora da máscara para manter o fundo limpo
    error_map = np.where(mask, error_map, np.nan)
    
    fig, (axe1,axe2,axe3) = plt.subplots(1, 3, figsize=(18, 5))
    v_min = 0
    v_max = max_error_viz
    # vmin=0 (Perfeição/Azul) e vmax=max_error_viz (Vermelho Máximo)
    # Qualquer erro igual ou maior que max_error_viz ficará no tom mais escuro de vermelho
    # Usaremos o colormap 'turbo' ou 'magma' para profundidade (percepção linear excelente)
    cmap_depth = 'turbo' 
    # Usaremos 'inferno' ou 'jet' para o erro (destaca o vermelho onde o erro é crítico)
    cmap_error = 'inferno' 

    # --- Plot 1: Ground Truth ---
    im0 = axe1.imshow(np.where(mask, depth_gt, np.nan), cmap=cmap_depth, vmax=max_dist)
    axe1.set_title("Ground Truth (LiDAR)", fontsize=14, fontweight='bold')
    axe1.axis('off') # Remove os eixos com números de pixels
    cbar0 = fig.colorbar(im0, ax=axe1, fraction=0.046, pad=0.04)
    cbar0.set_label('Metros', rotation=270, labelpad=15)

    # --- Plot 2: Predicted Depth ---
    im1 = axe2.imshow(np.where(mask, depth_pred, np.nan), cmap=cmap_depth, vmax=max_dist)
    axe2.set_title("Predicted Depth (Model)", fontsize=14, fontweight='bold')
    axe2.axis('off')
    cbar1 = fig.colorbar(im1, ax=axe2, fraction=0.046, pad=0.04)
    cbar1.set_label('Metros', rotation=270, labelpad=15)

    # --- Plot 3: Heatmap do MAE ---
    im2 = axe3.imshow(error_map, cmap=cmap_error, vmax=v_max)
    axe3.set_title(f"MAE Heatmap (Média: {np.nanmean(error_map):.4f}m)", fontsize=14, fontweight='bold')
    axe3.axis('off')
    cbar2 = fig.colorbar(im2, ax=axe3, fraction=0.046, pad=0.04)
    cbar2.set_label('Erro Absoluto (Metros)', rotation=270, labelpad=15)

    # Ajusta o layout para não cortar os títulos e salvar limpo
    plt.tight_layout()

    # ==============================================================================
    # 4. EXIBIR E SALVAR
    # ==============================================================================
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def decode_depth(depth_image_path:str, step:int=1):
    if depth_image_path.endswith(".png"):
        # 1. Carregar e aplicar subamostragem logo na leitura (ganho de performance)
        img = Image.open(depth_image_path).convert('RGB')
        data = np.array(img, dtype=np.float64)
        
        # Aplicar o step (pular linhas e colunas)
        data = data[::step, ::step]
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        
        # 2. Calcular Profundidade (Sua fórmula)
        depth = (r + g * 256.0 + b * 65536.0) / 10000.0
    elif depth_image_path.endswith(".npy"): # ends with npy
        data = np.load(depth_image_path)
        if len(data.shape)==3: data = np.squeeze(data, axis=0) # acontece com npy do nerf, ele é: (batch,h,w)
        depth = data[::step, ::step]
    elif depth_image_path.endswith(".tiff"):
        depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        depth = depth[::step, ::step]
    else: 
        depth = None
        print(f"Extension not found..")
    return depth