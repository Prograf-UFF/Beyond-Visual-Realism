import cv2, os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R_scipy
from .utils_plot import plot_depth_map, plot_cameras_3d, plot_convex_hull


def pontos_dentro_hull(pontos_a, pontos_b, margem=1.0, path_save=None): 
    """ margem em metros (pontos usados do worldPosition)
    """
    # Calcula a Envoltória Convexa
    hull = ConvexHull(pontos_a)
    
    # As equações do hull são do tipo Ax + By + C <= 0 para pontos dentro.
    # hull.equations retorna uma matriz onde cada linha é [A, B, C]
    # Scipy já normaliza os coeficientes (A² + B² = 1), 
    # então o resultado da equação é a distância direta do ponto à reta.
    equacoes = hull.equations
    
    # Verificamos cada ponto de B contra todas as equações da borda
    # Um ponto está "dentro + margem" se ele satisfaz Ax + By + C <= margem para TODAS as faces
    dentro = np.all(np.dot(equacoes[:, :-1], pontos_b.T) + equacoes[:, -1:] <= margem, axis=0)
    
    #plot debug
    if path_save:
        pontos_dentro = pontos_b[dentro]
        pontos_fora = pontos_b[~dentro]
        plot_convex_hull(hull_objeto=hull, lista_a=pontos_a, pontos_dentro=pontos_dentro, pontos_fora=pontos_fora, path_save=path_save)
    return dentro, hull

def generate_views(data:dict, image_processing_cubemap_theta_start: int=0,image_processing_cubemap_theta_end:int=360, image_processing_cubemap_theta_step:int=30, image_processing_cubemap_size:int=512,image_processing_cubemap_fov:int=90):
    nerf_views = list()
    # Geramos oss views para cada fotosfera
    for view_theta in range(image_processing_cubemap_theta_start,image_processing_cubemap_theta_end, image_processing_cubemap_theta_step):
        phi = 0. # Assumimos que todos estan en mesmo nivel
        nerf_views.append({
            'resource_id': os.path.basename(data['file'])[:-4], 
            'full_path': data['file'], 
            'from_lidar': data['from_lidar'], # usado só quando for trabalhar com lidar
            'worldPosition': data['worldPosition'],
            'phi': phi,           
            'theta': view_theta, 
            'image_processing_cubemap_size': image_processing_cubemap_size, 
            'image_processing_cubemap_fov':image_processing_cubemap_fov})
    return nerf_views

def get_camera_poses(images_txt_path):
    """
    Lê o arquivo images.txt e retorna uma lista de dicionários contendo
    o centro da câmera (C) e a matriz de rotação câmera->mundo (R_wc).
    """
    poses = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 10:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]
            
            # Rotação mundo -> câmera (R_cw)
            r = R_scipy.from_quat([qx, qy, qz, qw])
            R_cw = r.as_matrix()
            t_cw = np.array([tx, ty, tz])
            
            # Matriz de Rotação Câmera -> Mundo (R_wc = R_cw.T)
            R_wc = R_cw.T
            
            # Centro da Câmera no mundo em metros: C = -R_wc * t
            C_world = -R_wc @ t_cw
            
            poses.append({
                'name': image_name,
                'center': C_world,
                'R_wc': R_wc
            })
    #plot_cameras_3d(poses)
    return poses

def get_depth_and_mask(rgb_image_path:str, depth_file_path:str, method:str='eth3d'):
    if method=="RGBD".lower():
        pass # TODO
    elif method=="ETH3D".lower():
        rgb_img = cv2.imread(rgb_image_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_img.shape

        # 2. Leia o arquivo de profundidade como um binário float32 (4 bytes por pixel)
        with open(depth_file_path, "rb") as f:
            depth_raw = np.frombuffer(f.read(), dtype=np.float32)

        # 3. Redimensione a matriz de volta para a resolução da imagem (Row-Major)
        depth_in_meters = depth_raw.copy()
        depth_in_meters = depth_in_meters.reshape((height, width))

        # 3. Identifica onde estão os valores inválidos (Inf ou NaN)
        # Retorna uma matriz booleana (True onde for inválido, False onde for válido)
        invalid_mask = np.isinf(depth_in_meters) | np.isnan(depth_in_meters)

        # 4. Converte a máscara booleana para uma imagem em escala de cinza (uint8):
        # 255 (Branco) = Pixels Inválidos (Sem medição / Inf / NaN)
        #   0 (Preto)  = Pixels Válidos (Com profundidade real em metros)
        mask_image = (invalid_mask * 255).astype(np.uint8)
        # 4. Trate os pixels sem medição (locais sem retorno do laser ficam como Inf ou NaN)
        depth_in_meters[invalid_mask] = 0.0
    else:
        pass # TODO: implementar seu metodo
    
    #plot_depth_map(rgb_img, depth_in_meters, invalid_mask)
    return depth_in_meters, mask_image

#def main(colmap_images_path:str='images.txt'):
#    poses = get_camera_poses(colmap_images_path)
#    print(f"Total de câmeras carregadas: {len(poses)}")
#    plot_cameras_3d(poses)


