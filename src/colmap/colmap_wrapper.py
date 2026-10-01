import os
import pycolmap
import numpy as np


def run_bundle_adjuster(recon_path, ba_options: pycolmap.BundleAdjustmentOptions = None, ba_config: pycolmap.BundleAdjustmentConfig = None):
    #recon_path = ".tmp_work\\tmp2fpmfing\\sparse\\1"
    reconstruction = pycolmap.Reconstruction(recon_path)

    # Bundle Adjustment options (CLI-equivalent)
    if ba_options is None:
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_principal_point = True
        ba_options.refine_focal_length = True
        ba_options.refine_extra_params = True

    # Bundle Adjustment config
    if ba_config is None:
        ba_config = pycolmap.BundleAdjustmentConfig()

    ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)

    # By default, optimize all registered images and points
    for image_id, image in reconstruction.images.items():
        if image.has_pose:
            ba_config.add_image(image_id)

    # Run bundle adjuster
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, reconstruction)
    def run_ba(adjuster): adjuster.solve()   # NOTHING else here
    run_ba(bundle_adjuster)

    # Extract the covariances.
    cov_options = pycolmap.BACovarianceOptions(params=(pycolmap.BACovarianceOptionsParams.ALL))

    # Estimate covariance (requires bundle_adjuster)
    ba_covariance = pycolmap.estimate_ba_covariance(options=cov_options, reconstruction=reconstruction, bundle_adjuster=bundle_adjuster)

    exported_covs = {
        'poses': dict(),
        'points': dict(),
        'point_poses': list(),
        'image_names': dict()
    }

    def qvec2rotmat(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    for point3D_id, point3D in reconstruction.points3D.items():
        pto_cov = ba_covariance.get_point_cov(point3D_id)
        if pto_cov is None:
            continue

        if point3D_id not in exported_covs['points']:
            exported_covs['points'][point3D_id] = np.concatenate([point3D.xyz, np.asarray(pto_cov).flatten(), np.array([point3D.error, len(point3D.track.elements)])])

        for track_el in point3D.track.elements:
            image_id = track_el.image_id
            image = reconstruction.images[image_id]
            if not image.has_pose:
                continue
            if image_id not in exported_covs['poses']:
                # Retrieve pose covariance.
                qvec = image.frame.rig_from_world.rotation.quat
                tvec = image.frame.rig_from_world.translation
                R = qvec2rotmat(qvec)
                C = -R.T @ tvec
                pose_cov = ba_covariance.get_cam_cov_from_world(image_id)
                if pose_cov is None:
                    continue
                exported_covs['poses'][image_id] = np.concatenate([C, R.flatten(), np.asarray(pose_cov).flatten()])
                exported_covs['image_names'][image_id] = image.name
            # Save relationship between point and pose (for now, only the pixel coordinates of the observation).
            if track_el.point2D_idx >= len(image.points2D):
                continue
            d = np.linalg.norm(point3D.xyz - C)     # Distance from point to camera center to use as scale reference further.
            px, py = np.floor(image.points2D[track_el.point2D_idx].xy).astype(int)
            exported_covs['point_poses'].append(
                (point3D_id, image_id, d, px, py)
            )

    exported_covs['points'] = np.vstack([np.hstack([idx, xyz_cov]) for idx, xyz_cov in exported_covs['points'].items()])
    exported_covs['poses'] = np.vstack([np.hstack([idx, xyz_cov]) for idx, xyz_cov in exported_covs['poses'].items()])
    exported_covs['image_names'] = np.vstack([np.hstack([idx, xyz_cov]) for idx, xyz_cov in exported_covs['image_names'].items()])

    np.savez(os.path.join(os.path.abspath(recon_path), 'exported_covs.npz'), **exported_covs)
    reconstruction.write(recon_path)
