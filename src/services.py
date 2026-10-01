import random, os, json, shutil
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from tempfile import TemporaryDirectory
from .utils import get_camera_poses, pontos_dentro_hull, generate_views
from .utils_plot import plot_pose_comparison_2d
from .colmap import ColmapOdometry
from .metrics import Metrics

class Service(ABC):
    '''An abstract class to handle available services.'''

    def __init__(self, data_adapter=None, **kwargs):
        self.data_adapter = data_adapter

    def preprocess(self, **kwargs):
        return self.data_adapter.prepare_input(**kwargs)

    @abstractmethod
    def process(self, **kwargs):
        pass

    def postprocess(self, **kwargs):
        return self.data_adapter.prepare_output(**kwargs)
    
    def run(self, **kwargs):
        self.preprocess(**kwargs)
        output = self.process(**kwargs)
        self.postprocess(**kwargs)
        return output


class COLMAPService(Service):
    
    AVAILABLE_MODES: List[str] = ["RGBD", "ETH3D", "OTHER"]
    
    def __init__(self, **kwargs):
        super(COLMAPService, self).__init__(**kwargs)

    @classmethod
    def get_frames_json(cls, dataset_mode:str, images_path:str, poses_filepath:str, workspace_path:str, margem_convex_hull:float, eth3d_sample_debug:int=4):
        """  """
        views_dict = dict()
        frames_json = None
        if dataset_mode.lower()=="RGBD".lower():
            pass # TODO # baseado em imagens equiretangulares (usar cubemaps info)
        elif dataset_mode.lower()=="ETH3D".lower():
            # get info (Wolrd Positions)
            poses = get_camera_poses(poses_filepath)
            random.seed(42)
            choose_random_images = random.sample(range(len(poses)), eth3d_sample_debug) # escolhemos aleatoriamente 'eth3d_sample_debug' imagens para testar o LiDAR sobre elas
            for idx, p in enumerate(poses):
                views_dict[p['name'][:-4]] = {'file': os.path.join(images_path, os.path.basename(p['name'])), 'worldPosition': list(p['center']), 'from_lidar': idx in choose_random_images}
            # Convex-Hull
            views_list = [views_dict[k] for k in views_dict.keys() if not views_dict[k]['from_lidar']]
            views_list_lidar = [views_dict[k] for k in views_dict.keys() if views_dict[k]['from_lidar']]
            list_2d_photos = np.array([k['worldPosition'][:2] for k in views_list])
            list_2d_lidar = np.array([k['worldPosition'][:2] for k in views_list_lidar])
            indices_dentro, _ = pontos_dentro_hull(pontos_a=list_2d_photos, pontos_b=list_2d_lidar, path_save=os.path.join(workspace_path, "debug_plot_convex_hull.png"), margem=margem_convex_hull)

            frames_normais = [generate_views(v, 0, 30, 30, None, None) for v in views_list if not v['from_lidar']]
            frames_lidar = [generate_views(v, 0, 30, 30, None, None) for v in np.array(views_list_lidar)[indices_dentro]]
            frames_json = np.array(frames_normais + frames_lidar).flatten()
        else:
            pass # TODO: implementar se for usado outro dataset

        assert frames_json is not None, f"Unable to obtain the Frames-JSON for dataset {dataset_mode}."
        # Save frames info
        frames_json_path = os.path.join(workspace_path, f'frames.json')
        with open(frames_json_path, 'w') as frames_json_file:
            json.dump(frames_json.tolist(), frames_json_file)
        return frames_json_path

    
    @classmethod
    def preprocess(cls, dataset_mode:str, frames_json_path:str, lidar_rgb_path:str, lidar_depth_path:str, images_path:str, output_path:str, poses_filepath:str, workspace_path_override: str, margem_convex_hull:float=1., image_processing_cubemap_theta_start: int=0,image_processing_cubemap_theta_end:int=360, image_processing_cubemap_theta_step:int=30, image_processing_cubemap_size:int=1536,image_processing_cubemap_fov:int=90, **kwargs) -> None:
        assert dataset_mode.upper() in COLMAPService.AVAILABLE_MODES

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(workspace_path_override, exist_ok=True)
        with TemporaryDirectory(dir=workspace_path_override) as workspace_path:
            if not os.path.exists(frames_json_path):
                frames_json_path = COLMAPService.get_frames_json(dataset_mode, images_path, poses_filepath, workspace_path, margem_convex_hull)

            with open(frames_json_path, 'r') as f:
                frames_json = json.load(f)

            colmap = ColmapOdometry(frames=frames_json, colmap_matcher='EXHAUSTIVE', workspace_root_path=workspace_path, **kwargs)
            colmap(dataset_mode=dataset_mode, **kwargs)

            # Copiar o workspace_path para a pasta odometry
            shutil.copytree(workspace_path, output_path, dirs_exist_ok=True)
            # gerar plot para comparar os resultados
            plot_pose_comparison_2d(poses_filepath, os.path.join(output_path, 'text/0/images.txt'), report_filepath=os.path.join(output_path, "pose_comparision.png"))
        pass


class MetricService(Service):
    
    def __init__(self, **kwargs):
        super(MetricService, self).__init__(**kwargs)

    @classmethod
    def get_metrics(cls, metrics_gt_path:str, metrics_predicted_path:str, metrics_output_path:str, metrics_max_dist:float=10, metrics_plot_mae:bool=False, **kwargs):
        compute_metrics = Metrics(metrics_gt_path=metrics_gt_path,
                    metrics_predicted_path=metrics_predicted_path,
                    metrics_output_path=metrics_output_path,
                    metrics_max_dist=metrics_max_dist,
                    metrics_plot_mae=metrics_plot_mae)
        compute_metrics()
