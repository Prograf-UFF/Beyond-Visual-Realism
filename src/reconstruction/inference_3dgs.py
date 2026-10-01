import os, re, cv2, json
import torch
import numpy as np
import torchvision
from argparse import Namespace
from typing import Tuple
from tqdm import tqdm

from gaussian_renderer import GaussianModel, render
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def cfg_to_dict(model_path):
    cfg_path = os.path.join(model_path, "cfg_args")
    with open(cfg_path, 'r') as f:
        content = f.read()
    # 1. Extrair apenas o que está dentro dos parênteses do Namespace(...)
    match = re.search(r"Namespace\((.*)\)", content)
    if not match: return {}

    inner_content = match.group(1)
    # 2. Regex para capturar pares chave=valor
    # Explicação da regex:
    # (\w+)           -> A chave (letras, números e _)
    # =               -> O sinal de igual
    # (               -> Início do grupo do valor
    #   '[^']*'       -> Opção A: Texto entre aspas simples
    #   |             -> OU
    #   [^,)]+        -> Opção B: Qualquer coisa até encontrar uma vírgula ou o parênteses final
    # )
    pattern = r"(\w+)=((?:'[^']*')|(?:[^,)]+))"
    pairs = re.findall(pattern, inner_content)

    config = {}
    for key, value in pairs:
        # Limpeza básica (remover aspas)
        value = value.strip()
        if value.startswith("'") and value.endswith("'"): value = value[1:-1]
        
        # Conversão de tipos
        if value.lower() == 'true': value = True
        elif value.lower() == 'false': value = False
        elif value.isdigit(): value = int(value)
        else:
            try: value = float(value)
            except ValueError: pass
        config[key] = value
    return config


def run_inference(*, model_path: str, 
                    intrinsic: torch.Tensor, 
                    extrinsic: torch.Tensor, 
                    image_size: Tuple[int,int], # [heigth, width]
                    output_size: int, 
                    gs_data_device:str="cuda",
                    znear:float=0.01, zfar: float=100., **kwargs) -> torch.Tensor:
        
        params = cfg_to_dict(model_path)
        gaussians = GaussianModel(params.get('sh_degree', 3))
        
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device=gs_data_device)

        extrinsic[:, -2] *= -1
        R = np.array(extrinsic)[:3, [1,0,2]]
        T = np.array(extrinsic)[:3, -1] 
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        h, w = image_size #//params.get('resolution',  1.)

        # Montar a matriz C2W 4x4 corrigida
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        # Inverter para obter a W2C
        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3].T # Precisa da transposta
        T_w2c = w2c[:3, 3]

        # O 3DGS usa essa função para criar a world_view_transform
        # Note: R_w2c e T_w2c devem ser transpostos conforme a necessidade do CUDA do GS
        matrix_w2c = getWorld2View2(R_w2c, T_w2c)

        # Se for usar no renderizador vanilla:
        world_view_transform = torch.tensor(matrix_w2c).transpose(0, 1).to(gs_data_device)
        
        # Calcular FOV a partir da focal (necessário para a matriz de projeção)
        fovx = focal2fov(fx,w)
        fovy = focal2fov(fy,h)

        # Criar matriz de projeção
        proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        full_proj = world_view_transform @ proj.T.to(gs_data_device)
        
        # Acho que esse aqui é muito melhor 
        view = MiniCam(width = w//params.get('resolution',  1.), 
                       height = h//params.get('resolution',  1.),
                       fovx=fovx, fovy=fovy,
                       znear=znear, zfar=zfar, 
                       world_view_transform=world_view_transform, 
                       full_proj_transform=full_proj)
        
        pipeline_args = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=params.get('antialiasing', False))
        with torch.no_grad(): # TODO: a inferencia é muito rapida, mas carregar o PLY demora aprox. 1seg
            gaussians.load_ply(os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")) 
            render_out = render(view, gaussians, pipeline_args, background) 

        resize = torchvision.transforms.Resize([output_size, output_size]) # To RGB
        resize_nearest = torchvision.transforms.Resize([output_size, output_size], torchvision.transforms.InterpolationMode.NEAREST) # To others
        
        render_rgb = resize(render_out["render"].unsqueeze(0))[0]
        depth_map = resize_nearest(render_out["depth"].unsqueeze(0))[0]

        return render_rgb, depth_map


if __name__=="__main__":
    calibrated_poses_path ="data/setor1/calibrated_poses_lidar.json"
    nerfs_path = "output/depth_setor1_sh2"
    nerf_inference_frame_output_size = 1536
    nerf_inference_output_path = os.path.join(nerfs_path, f"render")

    with open(calibrated_poses_path, 'r') as f: 
        calibrated_poses = json.load(f)

    save_path_rgb = os.path.join(nerf_inference_output_path, "render_rgb")
    save_path_depth = os.path.join(nerf_inference_output_path, "render_depth")
    os.makedirs(save_path_depth, exist_ok=True)
    os.makedirs(save_path_rgb, exist_ok=True)

    valid_calibrated_poses = [v for v in calibrated_poses['views'] if v["should_be_discarded"]==False]
    for view in tqdm(valid_calibrated_poses, total=len(valid_calibrated_poses), desc="  Inference"):
            extrinsic = torch.Tensor(view['extrinsic'])
            intrinsic = torch.Tensor(view['intrinsic'])
            extrinsic[2, -1] -= calibrated_poses['nerf_floor_level']
            
            h, w = [view['image_processing_cubemap_size'], view['image_processing_cubemap_size']]
            rgb, disparidade_bruta = run_inference(model_path=nerfs_path, 
                                        image_size=(h, w),
                                        intrinsic=intrinsic, 
                                        extrinsic=extrinsic, 
                                        output_size=nerf_inference_frame_output_size)
            
            nerf_inference_name = view['view_filename'][:-4]
            # save render rgb
            torchvision.utils.save_image(rgb/rgb.max(), os.path.join(save_path_rgb, f'{nerf_inference_name}.png'))

            # save depth
            # TODO: averiguar -- no codigo novo retorna a inversa da profundidade (disparidade)
            # Substitui valores zero ou negativos por um valor mínimo muito pequeno (ex: 1e-6)
            disparidade_segura = torch.clamp(disparidade_bruta, min=1e-6)
            depth = 1.0 / disparidade_segura
            np.save(os.path.join(save_path_depth, nerf_inference_name+'.npy'), depth.detach().cpu().numpy())
            torchvision.utils.save_image(depth/depth.max(), os.path.join(save_path_depth, f'{nerf_inference_name}.png'))