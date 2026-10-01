import argparse, time, shutil
from src import services as algo

DATA_FOLDER = 'data'
OUTPU_FOLDER = 'output'
AVAILABLE_SERVICES = ['help', 'colmap_service', 'compute_metrics']
AVAILABLE_DATASET_MODES = ['ETH3D', 'RGBD', 'OTHER']

def main(args: argparse.Namespace):
    '''The main method.'''
    print('================================================================')
    print(f'   SERVICE: {args.module.upper()}                  ')
    print(f'============================================================\n')
    start = time.time()

    match args.module:
        case 'help':
            print('Run helper.')
        case 'colmap_service':
            algo.COLMAPService.preprocess(**vars(args))
        case 'compute_metrics':
            algo.MetricService.get_metrics(**vars(args))
        case _:
            raise ValueError(f'Command error. Choose one of the available services: {AVAILABLE_SERVICES}')

    # End process.
    print('\n')
    print(f'Time: {time.time() - start} s')
    print('Success!')
    print('============================================================')


if __name__ == "__main__":
    #get_depth_and_mask("data/pipes/images/dslr_images/DSC_0647.JPG", "data/pipes/ground_truth_depth/dslr_images/DSC_0647.JPG")
    #poses = get_camera_poses("data/pipes/dslr_calibration_jpg/images.txt")

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('module')
    group.add_argument('--data_path', type=str, required=False, default=DATA_FOLDER, help=f'Data path.')
    group.add_argument('--module', type=str, required=False, choices=AVAILABLE_SERVICES, default='help', help=f'Choose one of the available services: {AVAILABLE_SERVICES}')
    group.add_argument('--workspace_path_override', type=str, required=False, default=None, help=f'If not chosen, a temp. directory will be created and discarded at the end.')
    group = parser.add_argument_group('colmap')
    group.add_argument('--dataset_mode', type=str, required=False, default='eth3d', help=f'Choose one of the available dataset-modes: {AVAILABLE_DATASET_MODES}')
    group.add_argument('--frames_json_path', type=str, required=False, default='', help=f'Dictionary in corret format to process ColmapService.')
    group.add_argument('--poses_filepath', type=str, required=False, default='', help=f'Poses path in meters (brute file sparse-colmap).')
    group.add_argument('--lidar_depth_path', type=str, required=False, default='', help=f'LiDAR Depth-map path.')
    group.add_argument('--lidar_rgb_path', type=str, required=False, default='', help=f'LiDAR RGB path.')
    group.add_argument('--images_path', type=str, required=False, default='', help=f'Images RGB path.')
    group.add_argument('--output_path', type=str, required=False, default='', help=f'Odometry output path.')
    group.add_argument('--margem_convex_hull', type=float, required=False, default=1., help=f'Margem do Convex-Hull in meters.')
    group.add_argument('--colmap_bin_path', type=str, required=False, default=shutil.which('colmap'), help=f'Full path of the colmap binary.')
    group.add_argument('--colmap_camera_model', type=str, required=False, default='OPENCV', help=f'Camera model to process frames.')
    group.add_argument('--colmap_camera_params', type=str, required=False, default='', help=f'Camera parameters to process frames.')
    # Metrics module arguments.
    group = group.add_argument_group('metrics')
    group.add_argument('--metrics_gt_path', type=str, required=False, default=None, help=f'Depth path in RGBD format.')
    group.add_argument('--metrics_predicted_path', type=str, required=False, default=None, help=f'Depth inference path (*.npy or *.tiff).')
    group.add_argument('--metrics_output_path', type=str, required=False, default=None, help=f'Path to save the results.')
    group.add_argument('--metrics_max_dist', type=str, required=False, default=None, help=f'Maximum distance value for generating the clipping masks.')
    group.add_argument('--metrics_plot_mae', action='store_true', required=False, help=f'Save the results plot (MAE).')

    args = parser.parse_args()
    main(args)