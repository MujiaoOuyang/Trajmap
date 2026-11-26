"""Author's semantic fusion code with explicit runtime inputs.

Projection, arrow sampling, queue synchronization and class colors are retained.
See README.md for study calibration and external model requirements.
"""

import numpy as np
import copy
import json
import math
from pathlib import Path

if __package__:
    from .arrow_postprocess import extract_arrows_from_segmentation
    from .predict import get_colors
else:
    from arrow_postprocess import extract_arrows_from_segmentation
    from predict import get_colors

# No calibration is selected implicitly, including for helper API calls.
K = None
extrinsic_r2c = None
extrinsic_r2b = None


def configure_calibration(path):
    """Load the original matrix conventions from an explicitly chosen JSON file."""
    global K, extrinsic_r2c, extrinsic_r2b
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    intrinsic = np.asarray(config["intrinsic"], dtype=float)
    camera = np.asarray(config["extrinsic_r2c"], dtype=float)
    body = config.get("extrinsic_r2b")
    body = None if body is None else np.asarray(body, dtype=float)
    for name, matrix, shape in (("intrinsic", intrinsic, (3, 3)),
                                ("extrinsic_r2c", camera, (4, 4))):
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(name + " must be a finite " + str(shape) + " matrix")
        if np.linalg.matrix_rank(matrix) != shape[0]:
            raise ValueError(name + " must be invertible")
    if not np.allclose(camera[3], [0, 0, 0, 1]):
        raise ValueError("extrinsic_r2c must be a homogeneous transform")
    if body is not None and (body.shape != (4, 4) or not np.isfinite(body).all()):
        raise ValueError("extrinsic_r2b must be a finite (4, 4) matrix")
    K, extrinsic_r2c, extrinsic_r2b = intrinsic, camera, body
    return config


def get_data(file_path):
    file = open(file_path)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    data_list = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                 for line in lines if len(line) > 0 and line[0] != "#"]

    # for i in data_list:
    #     stamp = float(i[0])
    #     x = float(i[1])
    #     y = float(i[2])
    #     z = float(i[3])
    return data_list

def pcd_trans_44(pcd,tfmatrix44):
    pcd = pcd.astype('float32')
    length = len(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    print(f"transpcd shape: {transpcd.shape}, tfmatrix44 shape: {tfmatrix44.shape}")
    pcd[:3] = np.dot(tfmatrix44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd

# 把激光雷达点云投射到图像后，图像上的像素就获得了深度信息
def pcl2image(pc,w,h,ex):
    #pc is a np array
    #Transform to camera frame
    pc = pcd_trans_44(pc,ex)  # (65892, 5)
    # 过滤点云：保留 z > 0.2 米（前方）、x 在 [-10, 10] 米（左右范围）内的点
    d=pc[(pc[:,2]>0.2)&(pc[:,0]>-10)&(pc[:,0]<10)]
    e=d[:,:3]
    b = d[:, 3]
    # b = np.ones([len(e)]) * 10
    #Project to image
    r=K.dot(e.T)  # (3, 29441)
    print(f"pc shape: {pc.shape}, r shape: {r.shape}")
    scale = copy.deepcopy(r[2])
    r=r/r[2]
    t = r.T[:,:2]
    p = np.concatenate([t,b.reshape(-1,1),scale.reshape(-1,1)],axis=1)
    p = p[(p[:,0]>=0)&(p[:,1]>=1)&(p[:,0]<w)&(p[:,1]<h)]
    img = np.zeros((h,w,3))#his partintensity depth class
    #Next will use pybind11 to accelerate t
    for i in p:
        try:
            img[int(i[1]),int(i[0])]=np.array([i[2],i[3],0])
        except Exception as e:
            print(e,i[1],i[0],int(i[1]))
    return img
#Temporarily stop using interpolation, due to multi classes of the segmentation
    #interpolation
    interp_img = copy.deepcopy(img)
    pool = mp.Pool(24)
    per_epoch = int(img.shape[1] / 24)
    img_a = []
    for k in range(24):
        img_a.append((img[:,k * per_epoch:(k + 1) * per_epoch],k * per_epoch,(k + 1) * per_epoch))
    results = [pool.apply_async(interp_wrapper, args=(perimg[0],perimg[1],perimg[2])) for perimg in img_a]
    pool.close()
    pool.join()
    for res in results:
        perimg,start,end = res.get()
        interp_img[:,start:end]=perimg
    # for i in range(img.shape[1]):
    #     interp(img[:,i,1])
    return interp_img

def img2pcl(d_img):
    res = []
    sem = np.array(np.nonzero(d_img[:,:,1])).T
    for i in sem:
        col = d_img[tuple(i)]
        z=col[1]
        c=col[2]
        x=i[1]
        y = i[0]
        # y=d_img.shape[0]-i[0]
        res.append((x*z,y*z,z,c))
    if len(res) == 0:
        return np.array([]).reshape((0,4))
    res = np.array(res)
    e_i = np.matrix(extrinsic_r2c).I
    K_i = np.matrix(K).I
    src = K_i.dot(res[:,:3].T).T

    xyz = pcd_trans_44(src,e_i)
    res[:,:3]=xyz
    res=np.unique(res, axis=0)

    return res

def interpolate_depth_for_arrows(arrows_mask, depth_img, method='bilinear', ground_plane_fitting=True):
    """
    为箭头区域的所有像素插值深度信息
    
    Args:
        arrows_mask: 箭头区域的二值掩码 (H, W)
        depth_img: 深度图像，包含激光雷达投射的深度信息 (H, W, 3)
        method: 插值方法 ('bilinear', 'nearest', 'cubic')
        ground_plane_fitting: 是否使用地面平面拟合
    
    Returns:
        interpolated_depth: 插值后的深度图 (H, W)
    """
    from scipy import ndimage
    from scipy.interpolate import griddata
    from sklearn.linear_model import RANSACRegressor
    
    h, w = arrows_mask.shape
    interpolated_depth = np.zeros((h, w))
    
    # 获取有效深度点
    valid_depth_mask = depth_img[:, :, 1] > 0
    valid_coords = np.where(valid_depth_mask)
    valid_depths = depth_img[valid_coords[0], valid_coords[1], 1]
    
    if len(valid_depths) < 10:  # 深度点太少，无法插值
        return interpolated_depth
    
    # 方法1：基于周边点的插值
    if method == 'bilinear' or method == 'cubic':
        # 创建网格坐标
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        points = np.column_stack((valid_coords[0], valid_coords[1]))
        
        try:
            interpolated_depth = griddata(
                points, valid_depths, 
                (grid_y, grid_x), 
                method='linear' if method == 'bilinear' else 'cubic',
                fill_value=0
            )
        except:
            # 如果插值失败，使用最近邻
            interpolated_depth = griddata(
                points, valid_depths, 
                (grid_y, grid_x), 
                method='nearest',
                fill_value=0
            )
    
    elif method == 'nearest':
        # 最近邻插值
        distance_transform = ndimage.distance_transform_edt(~valid_depth_mask)
        indices = ndimage.distance_transform_edt(~valid_depth_mask, return_indices=True)[1]
        interpolated_depth = depth_img[:, :, 1][tuple(indices)]
    
    # 方法2：地面平面拟合（适用于道路场景）
    if ground_plane_fitting and len(valid_depths) > 100:
        try:
            # 将像素坐标转换为3D点
            K_inv = np.linalg.inv(K)
            camera_points = []
            
            for i in range(len(valid_coords[0])):
                y, x = valid_coords[0][i], valid_coords[1][i]
                depth = valid_depths[i]
                
                # 反投影到相机坐标系
                pixel_coord = np.array([x * depth, y * depth, depth])
                camera_coord = K_inv @ pixel_coord
                
                # 转换到激光雷达坐标系
                camera_point = np.array([camera_coord[0], camera_coord[1], camera_coord[2], 1.0])
                extrinsic_inv = np.linalg.inv(extrinsic_r2c)
                lidar_coord = extrinsic_inv @ camera_point
                
                camera_points.append([lidar_coord[0], lidar_coord[1], lidar_coord[2]])
            
            camera_points = np.array(camera_points)
            
            # 过滤地面附近的点（假设地面在z=0附近）
            ground_mask = (camera_points[:, 2] > -2) & (camera_points[:, 2] < 1)
            if np.sum(ground_mask) > 50:
                ground_points = camera_points[ground_mask]
                
                # 使用RANSAC拟合地面平面 ax + by + cz + d = 0
                X = ground_points[:, :2]  # x, y
                y = ground_points[:, 2]   # z
                
                ransac = RANSACRegressor(random_state=42, residual_threshold=0.1)
                ransac.fit(X, y)
                
                # 为箭头区域的像素估计地面深度
                arrow_coords = np.where(arrows_mask)
                for i in range(len(arrow_coords[0])):
                    img_y, img_x = arrow_coords[0][i], arrow_coords[1][i]
                    
                    if interpolated_depth[img_y, img_x] == 0:  # 只为没有深度的像素估计
                        # 使用周围有效深度的平均值作为初始估计
                        window_size = 20
                        y_min = max(0, img_y - window_size)
                        y_max = min(h, img_y + window_size)
                        x_min = max(0, img_x - window_size)
                        x_max = min(w, img_x + window_size)
                        
                        local_depths = depth_img[y_min:y_max, x_min:x_max, 1]
                        valid_local = local_depths[local_depths > 0]
                        
                        if len(valid_local) > 0:
                            estimated_depth = np.median(valid_local)
                            interpolated_depth[img_y, img_x] = estimated_depth
        
        except Exception as e:
            print(f"地面平面拟合失败: {e}")
    
    return interpolated_depth

def project_arrows_to_lidar(arrows_mask, depth_img, lidar_points, dense_sampling=True, sample_rate=0.3):
    """
    将提取的箭头区域投射到激光雷达点云上，支持密集采样
    
    Args:
        arrows_mask: 箭头区域的二值掩码 (H, W)
        depth_img: 深度图像，包含激光雷达投射的深度信息 (H, W, 3)
        lidar_points: 原始激光雷达点云 (N, 4) [x, y, z, intensity]
        dense_sampling: 是否进行密集采样
        sample_rate: 采样率，控制生成点云的密度
    
    Returns:
        arrow_points: 箭头区域对应的3D点云 (M, 4) [x, y, z, class_id=3]
    """
    arrow_points = []
    
    if dense_sampling:
        # 为箭头区域插值深度
        interpolated_depth = interpolate_depth_for_arrows(arrows_mask, depth_img)
        
        # 获取箭头区域的所有像素坐标
        arrow_coords = np.where(arrows_mask)
        
        # 根据采样率随机选择像素点
        total_pixels = len(arrow_coords[0])
        if total_pixels > 0:
            sample_count = max(1, int(total_pixels * sample_rate))
            indices = np.random.choice(total_pixels, sample_count, replace=False)
            
            selected_y = arrow_coords[0][indices]
            selected_x = arrow_coords[1][indices]
            
            K_inv = np.linalg.inv(K)
            extrinsic_inv = np.linalg.inv(extrinsic_r2c)
            
            for i in range(len(selected_y)):
                y, x = selected_y[i], selected_x[i]
                
                # 优先使用原始深度，其次使用插值深度
                depth = depth_img[y, x, 1] if depth_img[y, x, 1] > 0 else interpolated_depth[y, x]
                
                if depth > 0.1:  # 确保深度合理
                    # 反投影到3D空间
                    pixel_coord = np.array([x * depth, y * depth, depth])
                    camera_coord = K_inv @ pixel_coord
                    
                    # 转换到激光雷达坐标系
                    camera_point = np.array([camera_coord[0], camera_coord[1], camera_coord[2], 1.0])
                    lidar_coord = extrinsic_inv @ camera_point
                    
                    # 添加箭头点，类别标记为3
                    arrow_points.append([lidar_coord[0], lidar_coord[1], lidar_coord[2], 3])
    
    else:
        # 原始方法：只投射有激光点的像素
        arrow_pixels = np.where(arrows_mask)
        
        for i in range(len(arrow_pixels[0])):
            y, x = arrow_pixels[0][i], arrow_pixels[1][i]
            
            if depth_img[y, x, 1] > 0:
                depth = depth_img[y, x, 1]
                
                pixel_coord = np.array([x * depth, y * depth, depth])
                K_inv = np.linalg.inv(K)
                camera_coord = K_inv @ pixel_coord
                
                camera_point = np.array([camera_coord[0], camera_coord[1], camera_coord[2], 1.0])
                extrinsic_inv = np.linalg.inv(extrinsic_r2c)
                lidar_coord = extrinsic_inv @ camera_point
                
                arrow_points.append([lidar_coord[0], lidar_coord[1], lidar_coord[2], 3])
    
    if len(arrow_points) == 0:
        return np.array([]).reshape((0, 4))
    
    arrow_points = np.array(arrow_points)
    # 去除重复点（使用较小的容差）
    arrow_points = np.unique(np.round(arrow_points, decimals=2), axis=0)
    
    return arrow_points

color_classes = get_colors()

def color_convert(cls):
    c = color_classes[cls]
    return np.array([*c, 255]).astype(np.uint8).view('uint32')[0]

def pcd_trans(pcd,T,inverse=False):
    length = len(pcd)
    # 转置
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))  # 转成4*1
    # mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    # mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    # mat_r2e = np.dot(T, extrinsic_r2b)
    # if inverse:
    #     mat_r2e = np.matrix(mat_r2e).I
    pcd[:3] = np.dot(np.matrix(T), transpcd)[:3]  # 将激光雷达点云数据转到ENU坐标系下
    transedpcd = pcd.T
    return transedpcd

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Offline semantic LiDAR map generation from image/lidar/pose lists.")
    parser.add_argument("--data-root", required=True, help="Contains image/, lidar/, image_list.txt and lidar_list.txt")
    parser.add_argument("--pose-path", required=True, help="TUM poses: timestamp x y z qx qy qz qw")
    parser.add_argument("--output-pcd", required=True, help="New semantic PCD; existing paths are refused")
    parser.add_argument("--calibration", required=True, help="JSON with intrinsic and extrinsic_r2c matrices")
    parser.add_argument("--model-config", required=True, help="External MMSeg config, including its base configs")
    parser.add_argument("--weights", required=True, help="External WHU semantic checkpoint")
    parser.add_argument("--car-mask", required=True, help="Explicit NumPy mask matching the image resolution")
    parser.add_argument("--device", default="cuda", help="MMSeg device; initialized only when running")
    parser.add_argument("--start-time", type=float, required=True, help="Seconds; explicitly select the original time gate")
    parser.add_argument("--sync-threshold", type=float, default=0.05)
    parser.add_argument("--no-vis", action="store_true", default=True)
    parser.add_argument("--vis", action="store_false", dest="no_vis")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all aligned frames")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)
    if not math.isfinite(args.start_time) or not math.isfinite(args.sync_threshold) or args.sync_threshold <= 0:
        parser.error("start-time must be finite; sync-threshold must be positive and finite")
    if args.max_frames < 0:
        parser.error("max-frames must be >= 0")
    for path in (args.pose_path, args.calibration, args.model_config, args.weights, args.car_mask,
                 Path(args.data_root) / "image_list.txt", Path(args.data_root) / "lidar_list.txt"):
        if not Path(path).is_file():
            parser.error("Missing input file: " + str(path))
    for path in (args.output_pcd, args.log_file):
        if path and (Path(path).exists() or Path(path).is_symlink()):
            parser.error("Refusing to overwrite: " + str(path))
    if args.log_file and Path(args.log_file).resolve() == Path(args.output_pcd).resolve():
        parser.error("log-file and output-pcd must differ")
    configure_calibration(args.calibration)
    Path(args.output_pcd).parent.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        import contextlib
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_file, "x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                return run(args)
    return run(args)


def read_lidar_points(path):
    """Read the original six-field PCD contract using pypcd4's native API."""
    from pypcd4 import PointCloud

    pcd = PointCloud.from_path(path)
    expected_fields = ['x', 'y', 'z', 'intensity', 'timestamp', 'ring']
    data = {}
    for field in expected_fields:
        data[field] = pcd.pc_data[field]
    return np.stack((data['x'], data['y'], data['z'], data['intensity']), axis=1)


def run(args):
    import open3d as o3d
    import cv2
    import os
    from collections import deque
    from scipy.spatial.transform import Rotation
    from mmseg.apis import MMSegInferencer

    inferencer = MMSegInferencer(model=args.model_config, weights=args.weights, device=args.device)
    poseQueue = deque()
    imgQueue = deque()
    lidarQueue = deque()

    car_mask = np.load(args.car_mask, allow_pickle=False)

    data_root = args.data_root
    path_new = args.output_pcd

    pose_path = args.pose_path
    pose_data = get_data(pose_path)
    for i in range(len(pose_data)):
        poseQueue.appendleft(pose_data[i])

    imgs_path = os.path.join(data_root, "image_list.txt")
    for i in open(imgs_path, 'r').readlines():
        imgQueue.appendleft(i)

    lidars_path = os.path.join(data_root, "lidar_list.txt")
    for i in open(lidars_path, 'r').readlines():
        lidarQueue.appendleft(i)

    # 初始化 Open3D 可视化
    vis = None
    if not args.no_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Dynamic Point Cloud", width=1280, height=720)
    point_cloud = o3d.geometry.PointCloud()
    save_points = []  # [x, y, z, class]
    all_points = []
    all_colors = []
    first = True

    # 限制次数
    num = 0
    # threshold = 100
    while not len(imgQueue)==0 and not len(lidarQueue)==0 and not len(poseQueue)==0:
        if args.max_frames > 0 and num >= args.max_frames:
            break

        img_time = imgQueue.pop()
        lidar_time = lidarQueue.pop()
        print(f"img_time is {float(img_time[:-5])}, lidar_time is {lidar_time[:-5]}")
        if float(img_time[:-5])<float(lidar_time[:-5])-args.sync_threshold:
            lidarQueue.append(lidar_time)
            # continue
        elif float(img_time[:-5])>float(lidar_time[:-5])+args.sync_threshold:
            imgQueue.append(img_time)
            # continue
        else:
            pose = poseQueue.pop()
            pose_time = pose[0]
            print(f"pose time is {pose_time}")
            if float(pose_time)<float(lidar_time[:-5])-args.sync_threshold:
                lidarQueue.append(lidar_time)
                imgQueue.append(img_time)
                # continue
            elif float(pose_time)>float(lidar_time[:-5])+args.sync_threshold:
                poseQueue.append(pose)
                # continue
            else:
                print(f"pose time is {pose_time}")
                # if float(pose_time) < 1747899309.070192:
                #     continue
                if float(pose_time) < args.start_time:
                    continue
                # 已对齐
                rimg = cv2.imread(os.path.join(data_root, "image", img_time[:-1]))
                points = read_lidar_points(os.path.join(data_root, "lidar", lidar_time[:-1]))
                print(points.shape)

                src = pcl2image(points, rimg.shape[1], rimg.shape[0], extrinsic_r2c)
                #
                # segmentation
                result_image = inferencer(rimg, show=False, wait_time=0.001, return_vis=True)
                # print(result_image)
                seg_prediction = result_image['predictions'].astype(np.uint8)

                seg_prediction = seg_prediction*car_mask
                
                # 提取箭头区域
                target_mask = (seg_prediction == 3).astype(np.uint8) * 255
                arrows = extract_arrows_from_segmentation(result_image, car_mask=car_mask)
                
                # 创建箭头掩码并应用到分割结果
                arrows_mask = np.array(arrows).astype(np.bool_)
                
                # 保留原始道路标记，但将箭头区域单独标记
                seg_prediction_clean = seg_prediction * (~arrows_mask).astype(np.int32)
                seg_prediction = seg_prediction_clean + arrows_mask.astype(np.int32) * 3

                src[:, :, 2] = seg_prediction  # 将激光雷达点云数据映射到图像上
                
                # 使用新的箭头投射函数获取箭头点云（启用密集采样）
                arrow_points_3d = project_arrows_to_lidar(arrows_mask, src, points, 
                                                        dense_sampling=True, sample_rate=0.5)
                
                # 从图像获取语义点云
                sem_pcdata = img2pcl(src)
                
                # 合并箭头点云和其他语义点云
                if len(arrow_points_3d) > 0:
                    # 移除原有的箭头类别点（避免重复）
                    non_arrow_mask = sem_pcdata[:, 3] != 3
                    sem_pcdata_filtered = sem_pcdata[non_arrow_mask]
                    
                    # 合并箭头点云
                    if len(sem_pcdata_filtered) > 0:
                        sem_pcdata = np.vstack([sem_pcdata_filtered, arrow_points_3d])
                    else:
                        sem_pcdata = arrow_points_3d
                    
                    print(f"提取到 {len(arrow_points_3d)} 个箭头点云")
                    
                    # 可选：保存箭头深度图用于调试
                    if len(arrow_points_3d) > 0:
                        debug_save = False  # 设置为True来保存调试图像
                        if debug_save:
                            interpolated_depth = interpolate_depth_for_arrows(arrows_mask, src)
                            import cv2
                            # 创建可视化图像
                            debug_img = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
                            debug_img[:, :, 0] = (arrows_mask * 255).astype(np.uint8)  # 红色通道显示箭头掩码
                            debug_img[:, :, 1] = (interpolated_depth / np.max(interpolated_depth) * 255).astype(np.uint8)  # 绿色通道显示深度
                            cv2.imwrite(f"debug_arrow_depth_{pose_time}.png", debug_img)
                # sem_pcdata = points
                print('sem_pcdata is  ', sem_pcdata.shape)
                # mask = sem_pcdata[:, 3] != 0
                # road_pcd = sem_pcdata[mask]

                t = np.array([float(pose[1]), float(pose[2]), float(pose[3])])
                q = np.array([float(pose[4]), float(pose[5]), float(pose[6]), float(pose[7])])
                # q = np.array([float(pose[5]), float(pose[6]), float(pose[7]), float(pose[4])])
                R = Rotation.from_quat(q)
                T = np.eye(4)
                T[:3, :3] = R.as_matrix()
                T[:3, 3] = t

                sem_world_pcd = pcd_trans(sem_pcdata, T)

                # road_pcd = np.nonzero(sem_world_pcd[:, 3]).index()
                color_arr = []
                for i in range(sem_world_pcd.shape[0]):
                    color = (0, 0, 0)
                    if sem_world_pcd[i, 3]==1:
                        color = (0, 1, 0)
                    elif sem_world_pcd[i, 3]==2:
                        color = (1, 0, 0)
                    elif sem_world_pcd[i, 3]==3:  # 箭头
                        color = (1, 1, 0)
                    elif sem_world_pcd[i, 3] == 5:  # 停止线
                        color = (0, 0, 1)
                    # else:
                    #     continue
                    color_arr.append(color)

                save_points.append(sem_world_pcd)
                all_points.append(sem_world_pcd[:, :3])
                all_colors.append(color_arr)

                # 合并点云
                combined_points = np.vstack(all_points) if all_points else sem_pcdata[:, :3]
                combined_colors = np.vstack(all_colors) if all_colors else color_arr

                point_cloud.points = o3d.utility.Vector3dVector(combined_points[:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(combined_colors)

                print(sem_pcdata.shape)
                # o3d.visualization.draw_geometries([point_cloud])

                if vis is not None:
                    if first:
                        vis.add_geometry(point_cloud)
                        first = False
                    else:
                        vis.update_geometry(point_cloud)
                    # vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()

                    # 相机位置：传感器位置 + 高度偏移
                    camera_pos = t + np.array([0.0, 0.0, 50.0])
                    # 前视点：沿朝向向量前移 look_ahead 米
                    forward = R.as_matrix() @ np.array([10.0, 0.0, 0.0])
                    look_at = camera_pos + forward
                    # 上向量：世界坐标系的 z 轴（ENU）
                    up = np.array([0, 0, 1])

                    # 更新相机视角
                    view_control = vis.get_view_control()
                    view_control.set_lookat(look_at)
                    view_control.set_front(camera_pos - look_at)  # 朝向向量
                    view_control.set_up(up)
                # view_control.set_eye(camera_pos)

                num = num+1
                # if num>=threshold:
                #     break

    if vis is not None:
        vis.destroy_window()
    if not point_cloud.has_points():
        raise RuntimeError("No semantic points generated; check input timing and calibration")
    if Path(path_new).exists() or Path(path_new).is_symlink():
        raise FileExistsError("Refusing to overwrite: " + str(path_new))
    if not o3d.io.write_point_cloud(path_new, point_cloud):
        raise OSError("Failed to write semantic cloud: " + str(path_new))

    # color_arr = []
    # for i in range(sem_world_pcd.shape[0]):
    #     color = (0, 0, 0)
    #     if sem_world_pcd[i, 3]==1:
    #         color = (0, 1, 0)
    #     elif sem_world_pcd[i, 3]==2:
    #         color = (1, 0, 0)
    #     color_arr.append(color)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(sem_world_pcd[:, 0], sem_world_pcd[:, 1], sem_world_pcd[:, 2], c=color_arr, marker='.')
    # plt.show()


if __name__ == "__main__":
    main()
