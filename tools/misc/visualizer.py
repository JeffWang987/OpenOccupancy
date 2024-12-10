from xvfbwrapper import Xvfb
vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()

import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from nuscenes.nuscenes import NuScenes
import os
import glob
import imageio
from nuscenes.utils.splits import create_splits_scenes


classname_to_color = {  # RGB.
    0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}

def make_video_from_images(image_dir, save_video_path, fps=12):
    if isinstance(image_dir, str):
        image_names = os.listdir(image_dir)
        image_names.sort()
        image_dir = [os.path.join(image_dir, image_name) for image_name in image_names if image_name[-3:] in ['jpg', 'png']]
    assert isinstance(image_dir, list)
    save_mp4_filename = os.path.join(save_video_path, 'gt_vedio.mp4')
    with imageio.get_writer(save_mp4_filename, mode='I', fps=fps) as writer:
        for i in range(len(image_dir)):
            image = imageio.imread(image_dir[i])
            writer.append_data(image)

def make_video_from_imagesv2(image_dir, save_video_path, fps=12):
	gt_dir = os.path.join(image_dir, 'gt')
	pred_dir = os.path.join(image_dir, 'pred')
	gt_image_names = os.listdir(gt_dir)
	gt_image_names.sort()
	pred_image_names = os.listdir(pred_dir)
	pred_image_names.sort()

	gt_image_dir = [os.path.join(gt_dir, image_name) for image_name in gt_image_names if image_name[-3:] in ['jpg', 'png']]
	pred_image_dir = [os.path.join(pred_dir, image_name) for image_name in pred_image_names if image_name[-3:] in ['jpg', 'png']]
	save_mp4_filename = os.path.join(save_video_path, 'gt_pred_vedio.mp4')

	with imageio.get_writer(save_mp4_filename, mode='I', fps=fps) as writer:
		for i in range(len(gt_image_dir)):
			image_gt = imageio.imread(gt_image_dir[i])
			image_pred = imageio.imread(pred_image_dir[i])
			image = np.concatenate([image_gt, image_pred], axis=1)
			writer.append_data(image)


def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0):
	figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
	scene = figure.scene
	plt_plot = mlab.points3d(
		voxels[:, 0],
		voxels[:, 1],
		voxels[:, 2],
		voxels[:, 3],
		colormap="viridis",
		scale_factor=voxel_size - 0.05 * voxel_size,
		mode="cube",
		opacity=1.0,
		vmin=0,
		vmax=16,
	)
	# define your viewpoint https://anhquancao.github.io/blog/2022/how-to-define-viewpoint-programmatically-in-mayavi/
    # 0.2
    # scene.camera.position = [797.1825129240959, 722.2999942823683, 1093.0300087292317]
    # scene.camera.focal_point = [4.642500221729279, 255.50000977516174, 255.50000977516174]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.7707784057228109, -0.31169591730936624, -0.5556494438079416]
    # scene.camera.clipping_range = [687.8009559310524, 1947.0882525181303]
    # scene.camera.compute_view_plane_normal()
    # scene.render()

    # 0.8
	scene.camera.position = [201.79105756147231, 182.06481277731717, 199.84849642000432]
	scene.camera.focal_point = [124.15548284707876, 136.5253512085988, 150.33785237365439]
	scene.camera.view_angle = 30.0
	scene.camera.view_up = [0.6490726531122639, -0.4095893637890184, -0.641046990518413]
	scene.camera.clipping_range = [135.93224406174954, 433.4965064160177]
	scene.camera.compute_view_plane_normal()
	scene.render()

	custom_colormap(plt_plot)
	mlab.draw()
	# mlab.show()
	vis_path = os.path.join(vis_root, '{:0>4d}.png'.format(idx))
	mlab.savefig(filename=vis_path)

if __name__ == '__main__':
	# params
	nusc_root = '/mnt/cfs/algorithm/xiaofeng.wang/public_data/nuScenes'
	occ_gt_root = '/mnt/cfs/algorithm/public_data/nusc_occ/v0.0-small'
	occ_pred_root = '/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/occupancy/occupancy_jeff_branch/work_dirs/lss_occ_r50_multimodal_wo_rawocc_smallgt/visualization'
	vis_root = '/mnt/data-2/data/xiaofeng.wang/data/vis/nusc_occ'
	mode = 'val'
	scene_start = 0
	scene_end = 1

	# init
	splits = create_splits_scenes()
	splits = splits[mode]
	splits = splits[scene_start: scene_end]
	nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=True)
	os.makedirs(os.path.join(vis_root, 'gt'), exist_ok=True)
	os.makedirs(os.path.join(vis_root, 'pred'), exist_ok=True)
	os.makedirs(os.path.join(vis_root, 'pred_high'), exist_ok=True)
	os.makedirs(os.path.join(vis_root, 'pred_low'), exist_ok=True)

	# start
	for scene in nusc.scene:
		if scene['name'] not in splits:
			continue
		scene_token = scene['token']
		sample_token = scene['first_sample_token']
		lidar_token = nusc.get('sample', sample_token)['data']['LIDAR_TOP']

		vis_idx = 0
		while lidar_token is not '':
				
			lidar_data = nusc.get('sample_data', lidar_token)
			if not lidar_data['is_key_frame']:
				lidar_token = lidar_data['next']
				continue

			vis_idx += 1
			this_gt_path = os.path.join(occ_gt_root, 'scene_{}'.format(scene_token), 'occupancy', '{}.npy'.format(lidar_token))
			this_pred_path = os.path.join(occ_pred_root, 'scene_{}'.format(scene_token), '{}.npy'.format(lidar_token))
			this_pred_low_path = os.path.join(occ_pred_root, 'scene_{}'.format(scene_token), '{}_low.npy'.format(lidar_token))
			this_pred_high_path = os.path.join(occ_pred_root, 'scene_{}'.format(scene_token), '{}_high.npy'.format(lidar_token))
			voxels_gt = np.load(this_gt_path)
			voxels_pred = np.load(this_pred_path)
			voxels_pred_low = np.load(this_pred_low_path)
			voxels_pred_high = np.load(this_pred_high_path)
			draw(voxels_gt, vis_root=os.path.join(vis_root, 'gt'), idx=vis_idx)
			draw(voxels_pred, vis_root=os.path.join(vis_root, 'pred'), idx=vis_idx)
			draw(voxels_pred_low, vis_root=os.path.join(vis_root, 'pred_low'), idx=vis_idx)
			draw(voxels_pred_high, vis_root=os.path.join(vis_root, 'pred_high'), idx=vis_idx)

			lidar_token = lidar_data['next']


		make_video_from_imagesv2(vis_root, vis_root, fps=12)
	