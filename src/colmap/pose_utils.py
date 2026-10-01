import pandas as pd
import math, os, json, tqdm
import numpy as np
from scipy.stats import zscore
from typing import Tuple, TypedDict, List, Any
from . import colmap_read_model
from .colmap_wrapper import run_bundle_adjuster

CameraId = str
CameraModel = str
Point2d = Tuple[float, float]
Point3d = Tuple[float, float, float]
class Camera(TypedDict):
    id: CameraId
    model: CameraModel
    size: Point2d
    center: Point2d
    focal_length: Point2d
    angle: Point2d
    fov: Point2d
    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
class OdometryPoint(TypedDict):
    frame: Any
    location: Point3d
    orientation: Point3d    # a 3-D versor pointing the direction.
    camera: Camera

# ===================================================================================================================================================

def run_system_command(cmd_with_args: str):
    err = os.system(cmd_with_args)
    if err: raise RuntimeError(f"Odometry has failed after running {cmd_with_args}.")

def load_camera_from_file(filename):
    cameras = dict()
    cameras_df = pd.read_csv(filename, sep=' ', comment='#', header=None, names=['id', 'model', 'width', 'height', 'focal_length_hor', 'focal_length_ver', 'center_x', 'center_y', 'k1', 'k2', 'k3', 'k4'])
    for camera in cameras_df.to_dict(orient='records'):
        size = (int(camera['width']), int(camera['height']))
        center = (float(camera['center_x']), float(camera['center_y']))
        focal_length = (float(camera['focal_length_hor']), float(camera['focal_length_ver']))
        angle = (math.atan(size[0] / (2 * focal_length[0])) * 2, math.atan(size[1] / (2 * focal_length[1])) * 2)
        fov = (np.degrees(angle[0]), np.degrees(angle[1]))
        cameras[camera['id']] = Camera(
            id = camera['id'],
            model = camera['model'],
            size = size,
            center = center,
            angle = angle,
            fov = fov,
            focal_length = focal_length,
            k1 = float(camera['k1']),
            k2 = float(camera['k2']),
            k3 = float(camera['k3']),
            k4 = float(camera['k4']),
            p1 = 0,
            p2 = 0
        )
    return cameras

def filter_outliers(points, threshold=1):
    # Compute the z-scores for each dimension
    z_scores = np.abs(zscore(points, axis=0))
    
    # Identify points where all dimensions have z-scores below the threshold
    mask = np.all(z_scores < threshold, axis=1)
    
    # Return only the points that are not outliers
    return points[mask]

def fit_plane_svd(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]
    return normal, centroid

def get_points_in_dominant_plane(points, threshold=1e-2, iterations=100):
    points = np.asarray(points)
    best_inliers = []
    
    for _ in range(iterations):
        selected = np.random.choice(len(points), 3, replace=False)
        sample = points[selected]
        normal, point_on_plane = fit_plane_svd(sample)
        
        distances = np.abs((points - point_on_plane) @ normal) / np.linalg.norm(normal)
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            
            return np.array(best_inliers)

def poses_from_colmap(realdir, best_model=None):
    sparse_path = os.path.join(realdir, 'sparse')

    # Iterate through COLMAP models and take the biggest one.
    biggest_model_folder = get_biggest_model_id(sparse_path) if best_model is None else best_model
    
    camerasfile = os.path.join(realdir, f'sparse/{biggest_model_folder}/cameras.bin')
    camdata = colmap_read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, f'sparse/{biggest_model_folder}/images.bin')
    imdata = colmap_read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]

    # Remove images that don't belong to the main model (just ignore them).
    images_folder = os.path.join(realdir, 'images')
    #for img_filename in os.listdir(images_folder): os.remove(os.path.join(images_folder, img_filename)) if len(list(filter(lambda name: name.startswith(img_filename), names))) == 0 else None

    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, f'sparse/{biggest_model_folder}/points3D.bin')
    pts3d = colmap_read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, names

def normalize_colmap_poses(poses: np.ndarray, translate_to_origin=True):
        # nerf perform better around zero.
        model_translation = np.eye(4)
        if translate_to_origin:
            model_translation[:3, -1] = -np.mean(poses[:, 3], axis=1)

        all_positions = poses[:, 3].T
        all_distances = np.linalg.norm(all_positions[:, np.newaxis, :] - all_positions, axis=2)
        max_distance = np.max(all_distances)

        all_distances_ = np.copy(all_distances)
        all_distances_[np.where(np.floor(all_distances) == 0)] = np.inf
        median_min_proximity = np.median(np.amin(all_distances_, axis=0))
        std_min_proximity = np.std(np.amin(all_distances_, axis=0))
        median_overall = np.median(all_distances[np.triu_indices(all_distances.shape[0], k=0)])
        std_overall = np.std(all_distances[np.triu_indices(all_distances.shape[0], k=0)])
        aspect_ratio = (all_positions[:, 0].max() - all_positions[:, 0].min()) / (all_positions[:, 1].max() - all_positions[:, 1].min())

        odometry_statistics = {
            'n_photos': all_distances.shape[0],
            'max_distance': max_distance,
            'aspect_ratio_overall': aspect_ratio if aspect_ratio < 1 else 1/aspect_ratio,
            'median_min_proximity': median_min_proximity,
            'std_min_proximity': std_min_proximity,
            'var_min_proximity': std_min_proximity ** 2,
            'cv_min_proximity': std_min_proximity / median_min_proximity,
            'std_overall': std_overall,
            'var_overall': std_overall ** 2,
            'cv_overall': std_overall / median_overall
        }

        return model_translation, odometry_statistics

def map_colmap_positions_from_submodel(submodel: int, workspace_root_path:str, workspace_colmap_output_text:str, workspace_image_path:str, workspace_colmap_output:str, cameras, frames=None, colmap_bin_path:str=None, colmap_max_alignment_error: float=2.0, max_distance_to_discard_views=float('inf'), colmap_prepare_for_reconstruction:bool=True, odometry_max_location_error=None, dataset_method:str=None):
    # 2. Open images file and map each camera position.
    path = dict()
    images_df = pd.read_csv(os.path.join(workspace_colmap_output_text, str(submodel), "images.txt"), sep=' ', comment='#', header=None, names=['id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_id', 'filename'], skiprows=lambda row: row % 2 != 0)
    # needed transformation to keep COLMAP original coordinates.
    flip_matrix = np.array([
        [+1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, +1]
    ])
    # frames source path.
    for frame in tqdm.tqdm(images_df.to_dict(orient='records'), '   Getting camera positions from COLMAP space'):
        frame_id = frame['filename']
        frame_image_path = frame_id.replace('_', '/') if dataset_method.lower()=="RGBD".lower() else frame_id
        # retrieve rotation matrix.
        quaternions = -np.array([frame['qw'], frame['qx'], frame['qy'], frame['qz']])
        rotmat = np.array([
                [1 - 2*quaternions[2]**2 - 2*quaternions[3]**2, 2*quaternions[1]*quaternions[2] - 2*quaternions[0]*quaternions[3], 2*quaternions[3]*quaternions[1] + 2*quaternions[0]*quaternions[2]],
                [2*quaternions[1]*quaternions[2] + 2*quaternions[0]*quaternions[3], 1 - 2*quaternions[1]**2 - 2*quaternions[3]**2, 2*quaternions[2]*quaternions[3] - 2*quaternions[0]*quaternions[1]],
                [2*quaternions[3]*quaternions[1] - 2*quaternions[0]*quaternions[2], 2*quaternions[2]*quaternions[3] + 2*quaternions[0]*quaternions[1], 1 - 2*quaternions[1]**2 - 2*quaternions[2]**2]
        ])
        reference = np.array([0, 0, -1])
        coordinates = (-1) * rotmat.T @ np.array([frame['tx'], frame['ty'], frame['tz']])
        orientation = (-1) * rotmat.T @ reference
        orientation = orientation - coordinates
        model_matrix = np.linalg.inv(np.r_[np.c_[rotmat, np.array([frame['tx'], frame['ty'], frame['tz']]).T], np.array([[0, 0, 0, 1]])]) @ flip_matrix
        model_matrix[:3, :3] = np.array([[np.cos(np.pi), 0, -np.sin(np.pi)], [0, 1, 0], [np.sin(np.pi), 0, np.cos(np.pi)]]) @ model_matrix[:3, :3]

        path[frame_id] = (OdometryPoint(
            frame_path = frame_image_path,
            model_matrix = model_matrix.tolist(),
            camera = cameras[int(frame["camera_id"])],
            metadata = {'colmap_location': [frame['tx'], frame['ty'], frame['tz']], 'colmap_quaternions': [frame['qw'], frame['qx'], frame['qy'], frame['qz']]}
        ))

    model_rescale = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # colmap swaps 'y' and 'z', just undo that.

    if frames is not None:
        sorted_frames = dict(sorted(path.items()))
        unique_resources = set([frame.split('_')[1] for frame in list(sorted_frames.keys())])
        # assert len(unique_resources) > 2, 'Colmap submodel not valid, with less than 3 photospheres.' # Descomentar se usar fotosferas

        poses, pts3d, perm, imgs = poses_from_colmap(workspace_root_path, best_model=submodel)
        best_poses_ids = get_points_in_dominant_plane(poses[..., :][:, 3].T)

        all_photos = dict()
        lidar_rids = list(set([f['resource_id'] for f in frames if f['from_lidar']]))
        for pose_id in best_poses_ids:
            img_ = imgs[pose_id].split('_')[1] if dataset_method.lower()=="RGBD".lower() else imgs[pose_id][5:-4]
            #if img_ in lidar_rids: continue # ignoramos as fotosferas que vem do LiDAR
            if img_ not in all_photos:
                all_photos[img_] = list()
            all_photos[img_].append(poses[..., pose_id][:, 3].T)
        
        for img in all_photos.keys():
            old_n = len(all_photos[img])
            all_photos[img] = filter_outliers(np.vstack(all_photos[img]))
            new_n = len(all_photos[img])

        key_frames = sorted(all_photos.keys(), key=lambda x: len(all_photos[x]), reverse=True)[:(max(min(len(all_photos), 4), 3))]
        key_frames_ = list()
        real_pts = list()
        for frame in list(sorted_frames.keys()):
            frame_id = frame[:frame.find('_')]
            resource_id = frame[len(frame_id)+1:]
            resource_id = resource_id[:resource_id.find('_')] if dataset_method.lower()=="RGBD".lower() else resource_id[:-4]
            if resource_id in key_frames:
                frame_real_position = list(filter(lambda frame: frame['resource_id'].startswith(resource_id), frames))[0]
                real_pts.append(np.array(frame_real_position['worldPosition']))
                key_frames.remove(resource_id)
                key_frames_.append(resource_id)
        real_pts = np.stack(real_pts)

        with open(os.path.join(workspace_root_path, f'real_coords_{submodel}.txt'), 'w') as f:
            prime_z = np.mean(real_pts[:, 2])
            for idx, real_pto in enumerate(real_pts):
                image_filename = list(filter(lambda image_filename: image_filename.find(key_frames_[idx]) >= 0, os.listdir(workspace_image_path)))[0]
                f.write(f'{image_filename} {real_pto[0]} {real_pto[1]} {real_pto[2] - prime_z + 1}\n')
        
        print("     **COLMAP: Model-Aligner..")
        run_system_command(f'{colmap_bin_path} model_aligner --input_path {os.path.join(workspace_colmap_output, submodel)} --output_path {os.path.join(workspace_colmap_output, submodel)} --ref_images_path {os.path.join(workspace_root_path, f"real_coords_{submodel}.txt")} --ref_is_gps 0 --alignment_max_error {colmap_max_alignment_error} --log_to_stderr 0')

        # 4. run final bundle adjuster (via PyColmap to extract covariances).
        covariance_stats = run_bundle_adjuster(os.path.join(workspace_colmap_output, submodel))

        # 5. run image undistortion to convert OPENCV camera to PINHOLE.
        print("     **COLMAP: Image-Undistorter..")
        run_system_command(f'{colmap_bin_path} image_undistorter --image_path {workspace_image_path} --input_path {os.path.join(workspace_colmap_output, submodel)} --output_path {os.path.join(workspace_colmap_output, submodel)} --output_type COLMAP --log_to_stderr 0')

        # 5. run model converter to access metadata in text format .
        txt_path = os.path.join(workspace_colmap_output_text, submodel)
        os.makedirs(txt_path, exist_ok=True)
        print("     **COLMAP: Model-Converter TXT..")
        run_system_command(f'{colmap_bin_path} model_converter --input_path {os.path.join(workspace_colmap_output, submodel)} --output_path {txt_path}  --output_type TXT --log_to_stderr 0')

        assert not np.isnan(model_rescale).any(), 'There was an error during the calculation of the model matrix. Try running the odometry again.'
    
    if colmap_prepare_for_reconstruction:
        poses, pts3d, perm, imgs = poses_from_colmap(workspace_root_path, best_model=submodel)
        model_rescale, odometry_statistics = normalize_colmap_poses(poses, translate_to_origin=True)

        valid_frames = list()
        for idx, img_path in enumerate(sorted(os.listdir(workspace_image_path))):
            if img_path not in imgs:
                os.remove(os.path.join(workspace_image_path, img_path))
            else:
                valid_frames.append(idx)

        print('\nOdometry statistics:')
        print(odometry_statistics)

        # save json of used views.
        used_views = list()
        filtered_poses = list()
        filtered_perms = list()
        filtered_perms_lidar = list()
        if frames is not None:
            mean_error_location_adjustment = 0
            for idx, perm_idx in enumerate(perm):
                img_filename = imgs[perm_idx]
                pose = poses[..., perm_idx]
                height, width, focal_length = pose[:, -1]

                remapped_pose = model_rescale @ np.vstack((pose[:, :-1], np.array([[0, 0, 0, 1]])))
                #frame_original_location = list(filter(lambda frame: frame['resource_id'].startswith(img_filename.split('_')[1]) and ('frame_number' not in frame or frame['frame_number'] == int(first_frame_number)), self.frames))[0]
                original_frame = frames[valid_frames[idx]]
                original_location = np.array(original_frame['worldPosition']) - np.array([0, 0, prime_z - 1])   # Keep prime plane at z=1.

                # check proximity between original and calibrated (default: 1.5m max. distance).
                location_error = np.linalg.norm(pose[:2, 3] - original_location[:2])
                is_lidar = True if img_filename.split('_')[1] in lidar_rids else False
                if is_lidar:
                    should_be_discarded = (location_error > max_distance_to_discard_views)
                else:
                    z_error = np.abs(pose[2, 3] - original_location[2])
                    should_be_discarded = (location_error > max_distance_to_discard_views or z_error > .2)
                
                if not should_be_discarded:
                    mean_error_location_adjustment += location_error
                    filtered_poses.append(pose)
                else: # deveria ser descartado
                    pass # only debug
                pose[:, :-1] = remapped_pose[:3]
                
                if is_lidar: filtered_perms_lidar.append(perm_idx)
                else: filtered_perms.append(perm_idx)

                # Include floor level at the extrinsic matrix.
                extrinsic = (np.linalg.inv(model_rescale) @ np.vstack((pose[:, :-1], np.array([[0, 0, 0, 1]]))))
                extrinsic[2, -1] += prime_z

                used_views.append({
                    'submodel_id': submodel,
                    'view_filename': img_filename,
                    'resource_id': original_frame['resource_id'],
                    'theta': original_frame.get('theta', None),
                    'phi': original_frame.get('phi', None),
                    'image_processing_cubemap_size': original_frame.get('image_processing_cubemap_size', None),
                    'image_processing_cubemap_fov': original_frame.get('image_processing_cubemap_fov', None),
                    'magnetic_north_alignment': original_frame.get('magnetic_north_alignment', None),
                    'extrinsic': extrinsic.tolist(),
                    'intrinsic': [[focal_length, 0, height//2], [0, focal_length, width//2], [0, 0, 1]],
                    'location_error': location_error,
                    'z_error': z_error,
                    'should_be_discarded': bool(should_be_discarded),
                    'from_lidar': is_lidar
                })

            mean_error_location_adjustment /= len([frame for frame in used_views if not frame['should_be_discarded']])
            odometry_statistics['mean_location_error'] = mean_error_location_adjustment
            odometry_statistics['floor_level'] = prime_z
            odometry_statistics['covariance_stats'] = covariance_stats
            # Check if location error is too high.
            if mean_error_location_adjustment > 1:
                print('Warning: The location error is greater than 1m.')
                assert not bool(odometry_max_location_error) or odometry_max_location_error >= mean_error_location_adjustment, f'The location error in COLMAP ({odometry_max_location_error}) is greater than the expected ({mean_error_location_adjustment}).'
            print(f'Discarded views due to poor calibration: {poses.shape[-1] - len(filtered_poses)}')
        else:
            filtered_perms = perm

        with open(os.path.join(workspace_root_path, 'used_frames.json'), 'w') as frames_json:
            json.dump({
                'rescale': model_rescale.tolist(),
                'statistics': odometry_statistics,
                'floor_level': prime_z,
                'used_frames': [uv for uv in used_views if not uv['from_lidar']]
            }, frames_json)
        # Working with LiDAR  ------------------------------------------------
        with open(os.path.join(workspace_root_path, 'used_frames_lidar.json'), 'w') as frames_json:
            json.dump({
                'rescale': model_rescale.tolist(),
                #'statistics': odometry_statistics,
                'floor_level': prime_z,
                'used_frames': [uv for uv in used_views if uv['from_lidar']]
            }, frames_json)
        # Convert used-frames to calibrated-poses
        with open(os.path.join(workspace_root_path, 'used_frames_lidar.json'), 'r') as f:
            used_frames = json.load(f)
        selected_nerfs_json = dict()
        selected_nerfs_json['views'] = used_frames['used_frames']
        selected_nerfs_json['nerf_rescale'] = used_frames['rescale']
        selected_nerfs_json['nerf_floor_level'] = used_frames['floor_level']
        with open(os.path.join(workspace_root_path, 'calibrated_poses_lidar.json'), 'w') as f:
            json.dump(selected_nerfs_json, f)
        # -------------------------------------------------------------------
        # export poses bounds.
        poses_bounds_file_path = os.path.join(workspace_root_path, 'poses_bounds.npy')
        np.save(poses_bounds_file_path, np.array([np.concatenate([poses[..., i].ravel(), np.array([0, 0])], 0) for i in filtered_perms]))
        # Working with LiDAR
        poses_bounds_lidar_file_path = os.path.join(workspace_root_path, 'poses_bounds_lidar.npy')
        np.save(poses_bounds_lidar_file_path, np.array([np.concatenate([poses[..., i].ravel(), np.array([0, 0])], 0) for i in filtered_perms_lidar]))

        # 4.5 Remover visadas do Lidar ou should_be_discarted-- TODO (verificar porque las visadas do lidar estan marcadas como descartadas - ver distancia  error)
        # removemos as imagens que nao deveriam ser usadas do sparse
        discarted_images_sparse = os.path.join(workspace_root_path, 'discarted_images.txt')
        list_discarted_imgs = [x['view_filename'] for x in used_views if x['should_be_discarded'] or x['from_lidar']]
        submodel_sparse_folder = os.path.join(workspace_colmap_output_text, '0')
        txt_path = os.path.join(workspace_colmap_output_text, '0')
        with open(discarted_images_sparse, 'w') as f:
            for item in list_discarted_imgs:
                f.write(item + '\n')
        input_path = os.path.join(workspace_colmap_output, submodel)
        os.makedirs(submodel_sparse_folder, exist_ok=True)
        run_system_command(f'{colmap_bin_path} image_deleter --input_path {input_path} --image_names_path {discarted_images_sparse} --output_path {submodel_sparse_folder} --log_to_stderr 0')
        run_system_command(f'{colmap_bin_path} model_converter --input_path {input_path} --output_path {txt_path}  --output_type TXT --log_to_stderr 0')
        np.savez(os.path.join(workspace_root_path, 'model_back_from_colmap.npz'), model=model_rescale)

    return model_rescale


def get_biggest_model_id(sparse_path, allow_submodules=1):
    # Iterate through COLMAP models and take the bigger one.
    biggest_model_n_images = -1
    biggest_model_folder = None
    model_folders = os.listdir(sparse_path)
    if allow_submodules == 0 and len(model_folders) != 1:
        raise Exception('COLMAP was not capable to bundle all available frames in an unique model. You should try other parameters.')
    for model_folder in model_folders:
        model_path = os.path.join(sparse_path, model_folder)
        imagesfile = os.path.join(model_path, 'images.bin')
        imdata = colmap_read_model.read_images_binary(imagesfile)
        if len(imdata) > biggest_model_n_images:
            biggest_model_n_images = len(imdata)
            biggest_model_folder = model_folder
    return biggest_model_folder