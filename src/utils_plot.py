import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R_scipy

def load_camera_poses(images_txt_path):
    """
    Lê o arquivo images.txt e retorna um dicionário:
    {nome_da_imagem: {'center': np.array([X, Y, Z]), 'R_wc': Matriz_3x3}}
    """
    poses = {}
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                image_name = parts[9]
                
                # Rotação Mundo -> Câmera
                r = R_scipy.from_quat([qx, qy, qz, qw])
                R_cw = r.as_matrix()
                t_cw = np.array([tx, ty, tz])
                
                # Matriz de Rotação Câmera -> Mundo
                R_wc = R_cw.T
                
                # Centro focal em metros: C = -R_wc * t
                C_world = -R_wc @ t_cw
                
                poses[image_name] = {'center': C_world, 'R_wc': R_wc}
    return poses

def draw_camera_frustum_with_arrow(ax, center, R_wc, scale=0.1, color_side='blue', color_base='blue', color_arrow='blue'):
    """
    Desenha a câmera 3D com a base (lente) destacada e uma seta de direção.
    """
    X_w = R_wc[:, 0]
    Y_w = R_wc[:, 1]
    Z_w = R_wc[:, 2] # Vetor de visão (+Z)
    
    w, h = scale * 0.5, scale * 0.4
    base_center = center + scale * Z_w
    
    p1 = base_center - w * X_w - h * Y_w
    p2 = base_center + w * X_w - h * Y_w
    p3 = base_center + w * X_w + h * Y_w
    p4 = base_center - w * X_w + h * Y_w
    
    side_faces = [[center, p1, p2], [center, p2, p3], [center, p3, p4], [center, p4, p1]]
    base_face = [[p1, p2, p3, p4]]
    
    poly_sides = Poly3DCollection(side_faces, facecolors=color_side, linewidths=0.4, edgecolors='black', alpha=0.15)
    poly_base = Poly3DCollection(base_face, facecolors=color_base, linewidths=0.8, edgecolors='black', alpha=0.6)
    
    ax.add_collection3d(poly_sides)
    ax.add_collection3d(poly_base)
    
    ax.quiver(
        center[0], center[1], center[2],
        Z_w[0], Z_w[1], Z_w[2],
        length=scale * 1.5,
        color=color_arrow,
        linewidth=1.4,
        arrow_length_ratio=0.35,
        normalize=True
    )


def plot_cameras_3d(poses):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    centers = np.array([p['center'] for p in poses])
    X, Y, Z = centers[:, 0], centers[:, 1], centers[:, 2]
    
    # Define a escala visual da câmera com base nas dimensões da cena
    scene_extent = max(np.ptp(X), np.ptp(Y), np.ptp(Z))
    cam_scale = scene_extent * 0.05  # ~5% da escala total
    
    # Desenha cada câmera
    for pose in poses:
        draw_camera_frustum_with_arrow(ax, pose['center'], pose['R_wc'], scale=cam_scale)

    # Configurações de exibição
    ax.set_xlabel('X (metros)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (metros)', fontsize=11, labelpad=10)
    ax.set_zlabel('Z (metros)', fontsize=11, labelpad=10)
    ax.set_title('Posições e Direções das Câmeras (Seta = Vetor de Visão)', fontsize=13)
    
    # Mantém a proporção real dos eixos
    ax.set_xlim([X.min() - cam_scale, X.max() + cam_scale])
    ax.set_ylim([Y.min() - cam_scale, Y.max() + cam_scale])
    ax.set_zlim([Z.min() - cam_scale, Z.max() + cam_scale])
    ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])

    plt.tight_layout()
    plt.show()


def plot_depth_map(rgb_img, depth_matrix, invalid_mask, scale:float=0.9):
    depth_valid = depth_matrix.copy()
    depth_valid[invalid_mask] = np.nan

    rgb_img = cv2.resize(rgb_img, (0, 0), fx=scale, fy=scale)
    depth_valid = cv2.resize(depth_valid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    invalid_mask = cv2.resize(invalid_mask.astype(np.uint8), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)

    # Configura a paleta de cores 'turbo' para destacar os pixels inválidos em preto
    cmap_depth = plt.cm.get_cmap('turbo').copy()
    cmap_depth.set_bad(color='white')  # Pixels NaN ficam com a cor preta

    # ==========================================
    # Criando o Plot no Matplotlib (Grade 2x2)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Quadro 1: Imagem RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("1. Imagem RGB Original", fontsize=12)
    axes[0, 0].axis("off")

    # Quadro 2: Profundidade Bruta
    axes[0, 1].imshow(depth_matrix, cmap='gray')
    axes[0, 1].set_title("2. Profundidade Bruta (Inf / NaN = 0)", fontsize=12)
    axes[0, 1].axis("off")

    # Quadro 3: Máscara
    axes[1, 0].imshow(invalid_mask, cmap='gray')
    axes[1, 0].set_title("3. Máscara (Branco = Inf/NaN, Preto = Válido)", fontsize=12)
    axes[1, 0].axis("off")

    # Quadro 4: Profundidade Filtrada em Metros (Com barra de cores)
    im4 = axes[1, 1].imshow(depth_valid, cmap=cmap_depth, interpolation='nearest')
    axes[1, 1].set_title("4. Profundidade Filtrada (Metros)", fontsize=12)
    axes[1, 1].axis("off")

    # Adiciona a barra de legenda métrica (Colorbar) no último quadro
    cbar = fig.colorbar(im4, ax=axes[1, 1], shrink=0.6, aspect=30, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Distância (metros)', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_convex_hull(hull_objeto, lista_a, pontos_dentro, pontos_fora, path_save):
    # --- Visualização ---
    plt.figure(figsize=(8, 6))

    # Desenha a Envoltória (A)
    for simplex in hull_objeto.simplices:
        plt.plot(lista_a[simplex, 0], lista_a[simplex, 1], 'k-')

    plt.scatter(lista_a[:, 0], lista_a[:, 1], c='blue', alpha=0.3, label='Fotosferas Place-NeRFs')
    plt.scatter(pontos_dentro[:, 0], pontos_dentro[:, 1], c='green', marker='o', label='Fotosferas do LiDAR dentro de Place-NeRFs')
    plt.scatter(pontos_fora[:, 0], pontos_fora[:, 1], c='red', marker='x', label='Fotosferas do LiDAR fora de Place-NeRFs')

    plt.legend()
    plt.title("Verificação de Pontos dentro do Convex Hull")
    plt.savefig(path_save)
    plt.clf()


def plot_pose_comparison(eth3d_gt_path, colmap_aligned_path, report_filepath, nerf_floor_level):
    #nerf_floor_level = 0.18513638401877686
    print("--- Lendo arquivos de pose ---")
    gt_poses = load_camera_poses(eth3d_gt_path)
    colmap_poses = load_camera_poses(colmap_aligned_path)
    
    # Encontra imagens comuns a ambos os arquivos
    #common_names = sorted(list(set(gt_poses.keys()) & set(colmap_poses.keys())))
    colmap_names = list(colmap_poses.keys())
    gt_names = list(gt_poses.keys())
    
    #if not common_names:
    #    raise ValueError("Nenhuma imagem em comum foi encontrada entre os dois arquivos!")
        
    print(f"Câmeras correspondentes encontradas: {len(colmap_names)}")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    all_centers = []
    errors = []
    
    # 1. Desenha as linhas pontilhadas conectando o erro de cada par de câmera
    for name in colmap_names:
        indexes = [idx for idx, item in enumerate(gt_names) if name[5:-4] in item][0]
        name_gt = gt_names[indexes]
        gt_c = gt_poses[name_gt]['center']
        col_c = colmap_poses[name]['center']
        col_c[-1] -= nerf_floor_level
        
        all_centers.extend([gt_c, col_c])
        
        # Erro de posição em metros
        err = np.linalg.norm(gt_c - col_c)
        errors.append(err)
        
        # Desenha segmento de reta pontilhado cinza
        ax.plot([gt_c[0], col_c[0]], [gt_c[1], col_c[1]], [gt_c[2], col_c[2]], 
                color='gray', linestyle=':', linewidth=1.2, alpha=0.7)

    all_centers = np.array(all_centers)
    X, Y, Z = all_centers[:, 0], all_centers[:, 1], all_centers[:, 2]
    scene_extent = max(np.ptp(X), np.ptp(Y), np.ptp(Z))
    cam_scale = scene_extent * 0.04 # Escala visual das câmeras (~4% da cena)
    
    # 2. Desenha as câmeras do ETH3D GT (VERDE)
    for name in gt_names:
        draw_camera_frustum_with_arrow(ax, gt_poses[name]['center'], gt_poses[name]['R_wc'], 
                            scale=cam_scale, color_side='lightgreen', color_base='green', color_arrow='darkgreen')
        
    # 3. Desenha as câmeras do COLMAP ALINHADO (VERMELHO)
    for name in colmap_names:
        col_c = colmap_poses[name]['center']
        col_c[-1] -= nerf_floor_level
        draw_camera_frustum_with_arrow(ax, col_c, colmap_poses[name]['R_wc'], 
                            scale=cam_scale, color_side='cyan', color_base='red', color_arrow='crimson')

    # Cria elementos fictícios apenas para figurar na Legenda
    ax.plot([], [], [], color='green', marker='s', linestyle='None', label='ETH3D Ground Truth')
    ax.plot([], [], [], color='red', marker='s', linestyle='None', label='COLMAP Alinhado')
    ax.plot([], [], [], color='gray', linestyle=':', label='Desvio / Resíduo')

    # Configurações do gráfico
    ax.set_xlabel('X (metros)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (metros)', fontsize=11, labelpad=10)
    ax.set_zlabel('Z (metros)', fontsize=11, labelpad=10)
    ax.set_title('Comparação de Poses 3D: ETH3D GT vs. COLMAP Alinhado', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    
    # Proporções
    ax.set_xlim([X.min() - cam_scale, X.max() + cam_scale])
    ax.set_ylim([Y.min() - cam_scale, Y.max() + cam_scale])
    ax.set_zlim([Z.min() - cam_scale, Z.max() + cam_scale])
    ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
    
    # Estatísticas de Precisão
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    print("\n--- Relatório de Precisão Metrica ---")
    print(f"Erro Médio Absoluto (ATE): {np.mean(errors)*100:.2f} cm ({np.mean(errors):.4f} m)")
    print(f"Erro Quadrático Médio (RMSE): {rmse*100:.2f} cm ({rmse:.4f} m)")
    print(f"Erro Máximo de Posição:     {np.max(errors)*100:.2f} cm ({np.max(errors):.4f} m)")
    
    plt.tight_layout()
    plt.savefig(report_filepath, bbox_inches='tight', pad_inches=0.5)
    plt.clf()


def plot_pose_comparison_2d(eth3d_gt_path, colmap_aligned_path, report_filepath, plane='XY', show_labels=True):
    """
    Plota as posições 2D das câmeras.
    :param plane: 'XY' para vista superior (Top-Down) ou 'XZ' para vista lateral.
    """
    gt_poses = load_camera_poses(eth3d_gt_path)
    colmap_poses = load_camera_poses(colmap_aligned_path)
    
    #common_names = sorted(list(set(gt_poses.keys()) & set(colmap_poses.keys())))
    #if not common_names:
    #    raise ValueError("Nenhuma imagem em comum encontrada!")
    colmap_names = list(colmap_poses.keys())
    gt_names = list(gt_poses.keys())

    idx1, idx2 = (0, 1) if plane == 'XY' else (0, 2)
    axis_labels = ('X (metros)', 'Y (metros)') if plane == 'XY' else ('X (metros)', 'Z (metros)')

    fig, ax = plt.subplots(figsize=(11, 9))
    
    gt_coords, col_coords = [], []
    errors_2d = []

    colmap_by_gt = dict()
    for name in colmap_names:
        indexes = [idx for idx, item in enumerate(gt_names) if name[5:-4] in item][0]
        colmap_by_gt[name] = gt_names[indexes]
    # 1. Desenha as linhas pontilhadas de resíduo
    for name in colmap_names:
        gt_c = gt_poses[colmap_by_gt[name]]['center']
        col_c = colmap_poses[name]['center']

        # Posições 2D restritas ao plano atual
        gt_2d = gt_c[[idx1, idx2]]
        col_2d = col_c[[idx1, idx2]]
        
        gt_coords.append(gt_c)
        col_coords.append(col_c)
        
        # ERRO 2D NO PLANO SELECIONADO (ex: XY)
        err_2d = np.linalg.norm(gt_2d - col_2d)
        errors_2d.append(err_2d)
        
        ax.plot([gt_c[idx1], col_c[idx1]], [gt_c[idx2], col_c[idx2]], 
                color='gray', linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)

    gt_coords = np.array(gt_coords)
    col_coords = np.array(col_coords)
    errors_2d = np.array(errors_2d)
    
    # Estatísticas de erro no plano 2D
    mean_err_2d = np.mean(errors_2d)
    rmse_2d = np.sqrt(np.mean(errors_2d ** 2))
    max_err_2d = np.max(errors_2d)
    
    extent = max(np.ptp(gt_coords[:, idx1]), np.ptp(gt_coords[:, idx2]))
    arrow_scale = extent * 0.04

    # 2. Setas de Direção (ETH3D em Verde Escuro, COLMAP em Vermelho)
    for name in colmap_names:
        # ETH3D GT Seta
        c_gt = gt_poses[colmap_by_gt[name]]['center']
        v_gt = gt_poses[colmap_by_gt[name]]['R_wc'][:, 2][[idx1, idx2]]
        v_gt = v_gt / (np.linalg.norm(v_gt) + 1e-8) * arrow_scale
        ax.quiver(c_gt[idx1], c_gt[idx2], v_gt[0], v_gt[1], 
                  color='forestgreen', angles='xy', scale_units='xy', scale=1, 
                  width=0.004, alpha=0.6, zorder=2)
        
        # COLMAP Seta
        c_col = colmap_poses[name]['center']
        v_col = colmap_poses[name]['R_wc'][:, 2][[idx1, idx2]]
        v_col = v_col / (np.linalg.norm(v_col) + 1e-8) * arrow_scale
        ax.quiver(c_col[idx1], c_col[idx2], v_col[0], v_col[1], 
                  color='crimson', angles='xy', scale_units='xy', scale=1, 
                  width=0.004, alpha=0.6, zorder=2)

    # 3. MARCADORES CONCÊNTRICOS (Solução para sobreposição)
    # ETH3D: Argola maior e vazada por fora (Green Ring)
    ax.scatter(gt_coords[:, idx1], gt_coords[:, idx2], 
               facecolors='none', edgecolors='forestgreen', linewidths=2.0, s=120, 
               label='ETH3D GT (Argola)', zorder=4)

    # COLMAP: Ponto sólido menor por dentro (Red Dot)
    ax.scatter(col_coords[:, idx1], col_coords[:, idx2], 
               color='crimson', s=30, 
               label='COLMAP Alinhado (Ponto)', zorder=5)

    # 4. RÓTULOS OPCIONAIS (Numera cada câmera para fácil identificação)
    if show_labels:
        for idx, name in enumerate(colmap_names):
            gt_c = gt_poses[colmap_by_gt[name]]['center']
            # Adiciona pequeno deslocamento no texto para não cobrir o ponto
            ax.annotate(f"#{idx+1}", (gt_c[idx1], gt_c[idx2]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color='darkslategrey', alpha=0.85)

    # Legenda dos elementos do gráfico (no canto superior direito)
    ax.plot([], [], color='gray', linestyle=':', label='Desvio / Resíduo')
    ax.legend(loc='upper right', fontsize=10)

    # 5. CAIXA DE TEXTO COM O RELATÓRIO MÉTRICO 2D NO GRÁFICO
    report_text = (
        f"Precisão Métrica 2D (Plano {plane}):\n"
        f"• ATE Médio: {mean_err_2d * 100:.2f} cm ({mean_err_2d:.4f} m)\n"
        f"• RMSE 2D:   {rmse_2d * 100:.2f} cm ({rmse_2d:.4f} m)\n"
        f"• Erro Máx:  {max_err_2d * 100:.2f} cm ({max_err_2d:.4f} m)"
    )
    
    # Plota o texto no canto superior esquerdo dentro de uma caixa branca
    ax.text(
        0.03, 0.96, report_text,
        transform=ax.transAxes,
        fontsize=9.5,
        family='monospace',  # Fonte alinhada
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    # Configurações do gráfico
    ax.set_xlabel(axis_labels[0], fontsize=11)
    ax.set_ylabel(axis_labels[1], fontsize=11)
    ax.set_title(f'Comparação 2D de Poses no Plano {plane} (em Metros)', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(report_filepath, bbox_inches='tight', pad_inches=0.5)
    plt.clf()