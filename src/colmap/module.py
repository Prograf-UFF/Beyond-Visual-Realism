import numpy as np
import pandas as pd
import shutil, os, tqdm, math
from typing import List, Any
from datetime import datetime
from torch import nn
from .photo import get_clahe
from .pose_utils import load_camera_from_file, get_biggest_model_id, map_colmap_positions_from_submodel, run_system_command

VIEWS_IDX_STR_FORMAT = '%04d'
"""Formato de nomenclatura dos índices das VIEWS (index 4 = '0004')"""

DEFAULT_WORKSPACE_ROOT_PATH = '.odometry'

DEFAULT_WORKSPACE_IMAGE_PATH = 'images'
DEFAULT_WORKSPACE_MASK_PATH = 'masks'
DEFAULT_WORKSPACE_DATABASE_NAME = 'database.db'
DEFAULT_WORKSPACE_COLMAP_OUTPUT = 'sparse'
DEFAULT_WORKSPACE_COLMAP_OUTPUT_TEXT = 'text'

class ColmapOdometry(nn.Module):
    """Classe que implementa o método Colmap para realizar a calibração extrínseca."""
    def __init__(self, *, workspace_root_path: str=DEFAULT_WORKSPACE_ROOT_PATH, frames: List[Any]=None, colmap_bin_path=shutil.which('colmap'), colmap_camera_model: str='OPENCV', colmap_camera_params: str='', colmap_matcher: str='SEQUENTIAL', **kwargs):
        super(ColmapOdometry, self).__init__()
        
        if colmap_bin_path is None:
            raise ValueError('Could not find a COLMAP binary to run. Try to reinstall COLMAP or set "colmap_bin_path" in the configuration file. (ex.: colmap_bin_path=/usr/bin/colmap)')
        self.workspace_root_path = workspace_root_path
        self.colmap_bin_path = colmap_bin_path
        self.colmap_camera_model = colmap_camera_model.upper()
        self.colmap_camera_params = colmap_camera_params
        self.colmap_matcher = colmap_matcher.lower()
        self.frames = frames

        self.workspace_image_path = os.path.join(self.workspace_root_path, DEFAULT_WORKSPACE_IMAGE_PATH)
        self.workspace_database_file = os.path.join(self.workspace_root_path, DEFAULT_WORKSPACE_DATABASE_NAME)
        self.workspace_colmap_output = os.path.join(self.workspace_root_path, DEFAULT_WORKSPACE_COLMAP_OUTPUT)
        self.workspace_colmap_output_text = os.path.join(self.workspace_root_path, DEFAULT_WORKSPACE_COLMAP_OUTPUT_TEXT)
        self.workspace_mask_path = os.path.join(self.workspace_root_path, DEFAULT_WORKSPACE_MASK_PATH)
        
        os.makedirs(self.workspace_image_path, exist_ok=True)
        os.makedirs(self.workspace_colmap_output, exist_ok=True)
        os.makedirs(self.workspace_colmap_output_text, exist_ok=True)
        os.makedirs(self.workspace_mask_path, exist_ok=True)

    def forward(self, **kwargs):
        print(f'[{datetime.now()}] Processing COLMAP data')
        self.preprocessing(**kwargs)
        self.processing(**kwargs)
        print(f'[{datetime.now()}] Postprocessing COLMAP data')
        self.postprocessing(**kwargs)

    def preprocessing(self, *, image_processing_use_white_balance: bool=False, image_processing_use_masks:bool=False, dataset_mode:str=None, **kwargs):
        
        for idx, frame in tqdm.tqdm(enumerate(self.frames), '   Getting frames', total=len(self.frames)):
            resource = frame.get('source', frame)
            save_path= os.path.join(self.workspace_image_path, f'{VIEWS_IDX_STR_FORMAT % idx}_{resource["resource_id"]}')
            phi, theta = frame.get('phi', None), frame.get('theta', None)
            cubemap_size, cubemap_fov = frame.get('image_processing_cubemap_size', None), frame.get('image_processing_cubemap_fov', None)

            if np.abs(phi) == 90: theta = 0

            if dataset_mode=='RGBD'.lower():
                pass # TODO
            elif dataset_mode=='eth3d'.lower():
                current_frame_full_path = save_path+".png" 
                shutil.copy(resource['full_path'], current_frame_full_path)
                if image_processing_use_white_balance:
                    get_clahe(current_frame_full_path, save_path=current_frame_full_path)
                if image_processing_use_masks:
                    pass #TODO use sky and sea segmentation
            else:
                pass # TODO: gere seu propio pre-procesamento
            


    def processing(self, colmap_min_model_size: int=24, colmap_max_model_overlap: int=12, colmap_max_num_threads: int=8, colmap_max_num_features: int=4096, colmap_max_num_matches: int=5000, colmap_first_octave: int=-1, colmap_num_octaves: int=-1, colmap_octave_resolution: int=3, colmap_random_seed:int=0, **kwargs):
        # Default parameters consider models with, at least, the equivalent of 2 photospheres (30 degree views) size
        # and, at least, the equivalent of 1 photosphere overlap to merge them.
        # This policy favours larger models, but increases the risk of merging unacurate models.
        # To mitigate that risk, the user should tighten clean parameters 'max_distance_to_discard_views' and 'odometry_max_location_error'.
        progress_bar = tqdm.tqdm(range(5), '   Extracting features from images')

        # 1. run feature extractor.
        print(f'[{datetime.now()}] COLMAP: Feature Extraction')
        command_mask_path = f"--ImageReader.mask_path {self.workspace_mask_path}" if len(os.listdir(self.workspace_image_path)) == len(os.listdir(self.workspace_mask_path)) else ""
        run_system_command(f'{self.colmap_bin_path} feature_extractor --ImageReader.camera_model {self.colmap_camera_model} --ImageReader.camera_params "{self.colmap_camera_params}" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --SiftExtraction.max_num_features {colmap_max_num_features} --SiftExtraction.num_threads {colmap_max_num_threads} --SiftExtraction.first_octave {colmap_first_octave} --SiftExtraction.num_octaves {colmap_num_octaves} --SiftExtraction.octave_resolution {colmap_octave_resolution} --ImageReader.single_camera 1 --image_path {self.workspace_image_path} --database_path {self.workspace_database_file} {command_mask_path} --random_seed {colmap_random_seed} --log_to_stderr 0')
        # ex.: colmap feature_extractor --ImageReader.camera_model OPENCV_FISHEYE --ImageReader.camera_params "180,90,0,0,1,1,1,1" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --image_path frames_fisheye_180_90_bf --database_path database.db

        progress_bar.update(1)
        progress_bar.set_description(f'   Matching features from images ({self.colmap_matcher} method)')
        progress_bar.refresh()

        # 2. run matcher.
        print(f'[{datetime.now()}] COLMAP: Feature Matching')
        run_system_command(f'{self.colmap_bin_path} {self.colmap_matcher}_matcher --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches {colmap_max_num_matches} --SiftMatching.num_threads {colmap_max_num_threads} --database_path {self.workspace_database_file} --random_seed {colmap_random_seed} --log_to_stderr 0')
        # ex.: colmap sequential_matcher --SiftMatching.guided_matching=true --database_path database.db

        progress_bar.update(1)
        progress_bar.set_description(f'   Mapping models for the given cameras [may take some time]')
        progress_bar.refresh()

        # 3. run mapper.
        print(f'[{datetime.now()}] COLMAP: Mapper')
        run_system_command(f'{self.colmap_bin_path} mapper --image_path {self.workspace_image_path} --database_path {self.workspace_database_file} --output_path {self.workspace_colmap_output} --Mapper.min_model_size {colmap_min_model_size} --Mapper.max_model_overlap {colmap_max_model_overlap} --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point 1 --Mapper.ba_refine_extra_params 1 Mapper.ba_use_gpu 1 --Mapper.num_threads {colmap_max_num_threads} --random_seed {colmap_random_seed} --log_to_stderr 0')
        # ex.: colmap mapper --Mapper.multiple_models=0 --database_path database.db --image_path frames_fisheye_180_90_bf --output_path sparse/

        progress_bar.update(1)
        progress_bar.set_description(f'   Fine-adjusting models generated [may take some time]')
        progress_bar.refresh()

        # get biggest colmap submodel, i.e., that one that clusterizes the most frames.
        all_submodels = os.listdir(self.workspace_colmap_output)

        progress_bar.update(1)
        progress_bar.set_description(f'   Converting binary files to easy-readable TXT files')
        progress_bar.refresh()

        for submodel in all_submodels:
            # 5. run model converter to access metadata in text format .
            txt_path = os.path.join(self.workspace_colmap_output_text, submodel)
            os.makedirs(txt_path, exist_ok=True)
            run_system_command(f'{self.colmap_bin_path} model_converter --input_path {os.path.join(self.workspace_colmap_output, submodel)} --output_path {txt_path}  --output_type TXT --random_seed {colmap_random_seed} --log_to_stderr 0')

        progress_bar.update(1)
        progress_bar.set_description(f'   Calculating extrinsic parameters')
        progress_bar.refresh()

    def postprocessing(self, odometry_output_path: str=None, colmap_max_alignment_error: float=2.0, max_distance_to_discard_views=float('inf'), odometry_max_location_error=None,dataset_mode:str=None, **kwargs):
        if odometry_output_path is None:
            odometry_output_path = self.workspace_root_path
        
        # 1. open cameras file and translate positions.
        cameras = load_camera_from_file(os.path.join(self.workspace_colmap_output_text, "0", "cameras.txt"))
        biggest_model = get_biggest_model_id(self.workspace_colmap_output, allow_submodules=True)
        map_colmap_positions_from_submodel(biggest_model, self.workspace_root_path, self.workspace_colmap_output_text, self.workspace_image_path, self.workspace_colmap_output, cameras, self.frames, self.colmap_bin_path, colmap_max_alignment_error, max_distance_to_discard_views, colmap_prepare_for_reconstruction=True, odometry_max_location_error=odometry_max_location_error, dataset_method=dataset_mode)
        