import cv2
import numpy as np
from .equirec_projection import FlexibleEquirecProjection

ANGLE_STR_FORMAT = '%03d'
"""Formato de nomenclatura das angulações (angle 73° = '073')"""


def get_image_as_cubemap(equirec_rgb_path: str, equirec_depth_path:str, equirec_mask_path:str, save_path_rgb: str, save_path_depth: str, save_path_mask: str, *, theta: float=0.0, phi: float=0.0, image_processing_cubemap_size: int=960, image_processing_cubemap_fov: float=90.0, north_alignment:np.ndarray=None, **kwargs):
    """Retorna o caminho para a imagem perspectiva para a direção indicada."""
    projector = FlexibleEquirecProjection(
        equirec_rgb_path=equirec_rgb_path,
        equirec_depth_path=equirec_depth_path,
        equirec_mask_path=equirec_mask_path,
        kernel_size_mask=5 # 5 é suficiente
    )
    face_rgb, face_d_rgb, face_mask = projector.generate_virtual_face(
            theta_center_deg=theta,
            cubemap_alignment=north_alignment, #TODO: por agora nao é necesario (np.eye(3))
            fov_h_deg=image_processing_cubemap_fov, 
            fov_v_deg=image_processing_cubemap_fov,
            face_w=image_processing_cubemap_size,
            face_h=image_processing_cubemap_size
        )
    output_rgb = f'{save_path_rgb}_face_cubemap_{image_processing_cubemap_size}_{ANGLE_STR_FORMAT % int(theta)}_{ANGLE_STR_FORMAT % int(phi)}.png'
    cv2.imwrite(output_rgb, face_rgb)
    if face_d_rgb: cv2.imwrite(f'{save_path_depth}_face_cubemap_{image_processing_cubemap_size}_{ANGLE_STR_FORMAT % int(theta)}_{ANGLE_STR_FORMAT % int(phi)}.png', face_d_rgb)
    if face_mask: cv2.imwrite(f'{save_path_mask}_face_cubemap_{image_processing_cubemap_size}_{ANGLE_STR_FORMAT % int(theta)}_{ANGLE_STR_FORMAT % int(phi)}.png', face_mask)
    return output_rgb

def get_clahe(img_path:str, save_path:str):
    img = cv2.imread(img_path)
    # 1. Converter para LAB para mexer apenas na luminosidade (L)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Aplicar CLAHE (Equalização de histograma adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # 3. Juntar e converter de volta
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, final_img)