import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes."""
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if fix_radius > 0:
        scale_factor = 1.0 / fix_radius
    else:
        scale_factor = 1.0 / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def _make_synthetic_pointcloud(c2ws, timestamps):
    points = []
    point_times = []
    offsets = [
        (0.0, 0.0),
        (-0.35, -0.20),
        (-0.35, 0.20),
        (0.35, -0.20),
        (0.35, 0.20),
        (0.0, -0.30),
        (0.0, 0.30),
        (-0.55, 0.0),
        (0.55, 0.0),
    ]
    depths = [4.0, 8.0, 12.0]

    for idx, c2w in enumerate(c2ws):
        origin = c2w[:3, 3]
        right = c2w[:3, 0]
        down = c2w[:3, 1]
        forward = c2w[:3, 2]
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            continue
        forward = forward / forward_norm
        for depth in depths:
            base = origin + forward * depth
            for off_x, off_y in offsets:
                points.append(base + right * off_x * depth + down * off_y * depth)
                point_times.append([timestamps[idx]])

    if not points:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    return np.asarray(points, dtype=np.float32), np.asarray(point_times, dtype=np.float32)


def _load_mask(mask_path):
    if mask_path is None or not os.path.exists(mask_path):
        return None
    mask = np.array(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask[..., None]


def _resolve_longtail_mask_path(source_path, mask_dirname, cam_idx, frame_name):
    frame_candidates = [frame_name]
    if isinstance(frame_name, str) and frame_name.isdigit():
        frame_candidates.extend([
            f"{int(frame_name):06d}",
            f"{int(frame_name):04d}",
        ])
    frame_candidates = list(dict.fromkeys(frame_candidates))

    candidates = []
    mask_root = os.path.join(source_path, mask_dirname)
    if os.path.isdir(mask_root):
        for candidate_frame in frame_candidates:
            candidates.extend([
                os.path.join(mask_root, f"{candidate_frame}_{cam_idx}.png"),
                os.path.join(mask_root, f"{candidate_frame}_image_{cam_idx}.png"),
                os.path.join(mask_root, f"{candidate_frame}_cam_{cam_idx}.png"),
                os.path.join(mask_root, f"{candidate_frame}_cam{cam_idx}.png"),
                os.path.join(mask_root, f"image_{cam_idx}", f"{candidate_frame}.png"),
                os.path.join(mask_root, str(cam_idx), f"{candidate_frame}.png"),
            ])

    for candidate_frame in frame_candidates:
        candidates.extend([
            os.path.join(source_path, f"longtail_mask{cam_idx}", f"{candidate_frame}.png"),
            os.path.join(source_path, f"lt_mask{cam_idx}", f"{candidate_frame}.png"),
            os.path.join(source_path, f"lt_masks{cam_idx}", f"{candidate_frame}.png"),
        ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def readWaymoInfo(args):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    points = []
    points_time = []

    first_frame = max(0, int(getattr(args, "start_frame", 0)))
    last_frame = int(getattr(args, "end_frame", -1))
    if last_frame < 0 or last_frame >= len(car_list):
        last_frame = len(car_list) - 1
    if first_frame > last_frame:
        raise ValueError(
            f"Invalid Waymo frame range: start_frame={first_frame}, end_frame={last_frame}, total_frames={len(car_list)}"
        )
    car_list = car_list[first_frame:last_frame + 1]

    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval * (frame_num - 1) / 2, args.frame_interval * (frame_num - 1) / 2]
    else:
        time_duration = args.time_duration

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt'))

        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3]
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4)
        lidar2cam = pad_poses(lidar2cam)

        cam2lidar = np.linalg.inv(lidar2cam)
        c2w = ego_pose @ cam2lidar
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []
        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:args.cam_num]:
            image_path = os.path.join(args.source_path, subdir, car_id + '.png')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.0
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:args.cam_num]:
            sky_data = np.array(Image.open(os.path.join(args.source_path, subdir, car_id + '.png')))
            sky_mask = sky_data > 0
            sky_masks.append(sky_mask.astype(np.float32))

        lt_masks = []
        lt_mask_confs = []
        for cam_idx in range(args.cam_num):
            lt_mask_path = _resolve_longtail_mask_path(
                args.source_path,
                getattr(args, 'lt_mask_dirname', 'lt_masks'),
                cam_idx,
                car_id,
            )
            lt_mask_conf = _load_mask(lt_mask_path)
            if lt_mask_conf is None:
                lt_masks.append(None)
                lt_mask_confs.append(None)
            else:
                lt_mask_confs.append(lt_mask_conf.astype(np.float32))
                lt_masks.append((lt_mask_conf > 0).astype(np.float32))

        if len(car_list) > 1:
            timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        else:
            timestamp = 0.5 * (time_duration[0] + time_duration[1])

        point_xyz = None
        velodyne_path = os.path.join(args.source_path, "velodyne", car_id + ".bin")
        if os.path.exists(velodyne_path) and os.path.getsize(velodyne_path) > 0:
            point = np.fromfile(velodyne_path, dtype=np.float32, count=-1)
            if point.size % 6 != 0:
                raise ValueError(f"Unexpected velodyne shape for {velodyne_path}: float_count={point.size}")
            point = point.reshape(-1, 6)
            point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
            point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ ego_pose.T)[:, :3]
            points.append(point_xyz_world)
            point_time = np.full_like(point_xyz_world[:, :1], timestamp)
            points_time.append(point_time)

        for j in range(args.cam_num):
            point_camera = None
            if point_xyz is not None:
                point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3]
            R = np.transpose(w2c[j, :3, :3])
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovX = FovY = -1.0
            cam_infos.append(CameraInfo(
                uid=idx * args.cam_num + j,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=images[j],
                image_path=image_paths[j],
                image_name=car_id,
                width=HWs[j][1],
                height=HWs[j][0],
                sky_mask=sky_masks[j],
                lt_mask=lt_masks[j],
                lt_mask_conf=lt_mask_confs[j],
                timestamp=timestamp,
                pointcloud_camera=point_camera,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            ))

        if args.debug_cuda:
            break

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])
        cam_info.T[:] = w2c[:3, 3]
        if cam_info.pointcloud_camera is not None:
            cam_info.pointcloud_camera[:] *= scale_factor

    if points:
        pointcloud = np.concatenate(points, axis=0)
        pointcloud_timestamp = np.concatenate(points_time, axis=0)
        pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    else:
        timestamps = np.asarray([cam_info.timestamp for cam_info in cam_infos], dtype=np.float32)
        pointcloud, pointcloud_timestamp = _make_synthetic_pointcloud(c2ws, timestamps)

    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None, time=pointcloud_timestamp)
    if len(car_list) > 1:
        time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)
    else:
        time_interval = max(float(args.frame_interval), 1e-3)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        time_interval=time_interval,
        time_duration=time_duration,
    )

    return scene_info
