import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import os
import argparse
import matplotlib.pyplot as plt
import copy

# 类别编号定义
LANE_SOLID_CLASS = 1   # 车道实线
LANE_DASHED_CLASS = 2  # 车道虚线
LANE_ARROW_CLASS = 3  # 箭头
STOP_LINE_CLASS = 5    # 停止线

# 停止线检测参数
MIN_STOP_LINE_POINTS = 50  # 最小停止线点数，用于过滤噪声
MIN_LANE_POINTS = 250       # 最小车道线点数，用于验证停止线
FRONT_CHECK_DIST = 8.0     # 前方检查距离（米）
MIN_SIDE_POINTS = 20        # 每侧最少停止线点数

def load_point_cloud_with_class(filepath):
    """
    加载带有类别信息的点云数据
    返回: Nx4的数组，最后一列为类别ID
    """
    pcd = o3d.io.read_point_cloud(filepath)
    if not pcd.has_points():
        print(f"错误: 在{filepath}中未找到点")
        return None
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        class_ids = np.array([color_to_class_id(c) for c in colors])
        points_with_class = np.hstack([points, class_ids.reshape(-1, 1)])
        return points_with_class
    else:
        print("警告：点云文件不包含颜色信息，仅返回xyz")
        return np.hstack([points, np.zeros((points.shape[0],1))])

def color_to_class_id(color):
    """
    将颜色转换为类别ID
    color: (3,) array, float, 范围[0,1]
    """
    if np.allclose(color, [0, 1, 0], atol=0.05):
        return 1  # 车道实线
    elif np.allclose(color, [1, 0, 0], atol=0.05):
        return 2  # 车道虚线
    elif np.allclose(color, [1, 1, 0], atol=0.05):
        return 3  # 箭头
    elif np.allclose(color, [0, 0, 1], atol=0.05):
        return 5  # 停止线
    else:
        return 0  # 其它类别

def filter_points_by_class(points, class_ids):
    """
    根据类别过滤点云
    points: Nx4, 最后一列为类别
    class_ids: 类别ID列表
    """
    return points[np.isin(points[:, 3], class_ids)]

def load_poses_tum(pose_path):
    """
    加载TUM格式的位姿文件
    """
    poses = []
    timestamps = []
    with open(pose_path, 'r') as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            vals = line.strip().split()
            if len(vals) == 8:
                timestamps.append(float(vals[0]))
                poses.append([float(x) for x in vals[1:]])
    return timestamps, poses

def extract_window_by_pose(points, pose, window_length=20, window_width=8):
    """
    根据位姿提取前方窗口内的点云
    points: Nx4 点云 (x, y, z, class_id)
    pose: [x, y, z, qx, qy, qz, qw] 车辆位姿
    window_length: 沿车辆前进方向的长度
    window_width: 垂直方向的宽度
    """
    center = np.array(pose[:3], dtype=float)
    quat = np.array(pose[3:7], dtype=float)
    R = Rotation.from_quat(quat)
    forward = R.apply([1, 0, 0])  # 车辆前进方向
    left = R.apply([0, 1, 0])     # 车辆左侧方向
    rel = points[:, :3] - center
    proj_forward = np.dot(rel, forward)
    proj_left = np.dot(rel, left)
    mask = (proj_forward > 0) & (proj_forward < window_length) & (np.abs(proj_left) < window_width/2)
    return points[mask]

def get_direction_from_pose(pose):
    """
    从位姿获取前进方向向量
    """
    quat = np.array(pose[3:7], dtype=float)
    R = Rotation.from_quat(quat)
    forward = R.apply([1, 0, 0])  # 车辆前进方向
    return forward / np.linalg.norm(forward)

def visualize_point_cloud(points_with_class, special_points=None, special_color=None):
    """
    可视化点云
    """
    geometries = []
    # 原始点云可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_with_class[:, :3])
    # 按类别着色
    colors = []
    for c in points_with_class[:, 3]:
        if c == LANE_SOLID_CLASS:
            colors.append([0, 1, 0])  # 绿色 - 实线
        elif c == LANE_DASHED_CLASS:
            colors.append([1, 0, 0])  # 红色 - 虚线
        elif c == LANE_ARROW_CLASS:
            colors.append([1, 1, 0])  # 黄色 - 箭头
        elif c == STOP_LINE_CLASS:
            colors.append([0, 0, 1])  # 蓝色 - 停止线
        else:
            colors.append([0.5, 0.5, 0.5])  # 灰色 - 其他
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    geometries.append(pcd)
    
    # 特殊点
    if special_points is not None and len(special_points) > 0:
        special_pcd = o3d.geometry.PointCloud()
        special_pcd.points = o3d.utility.Vector3dVector(special_points[:, :3])
        if special_color is not None:
            special_pcd.paint_uniform_color(special_color)
        geometries.append(special_pcd)
    
    o3d.visualization.draw_geometries(geometries)

def save_point_cloud(points, filename, colors=None):
    """
    保存点云到文件
    """
    if len(points) == 0:
        print(f"警告：没有点可保存到 {filename}")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"已保存点云到: {filename}")

def is_valid_stop_line_simple(stop_line_points, window_points, pose, full_points=None):
    """
    简化的停止线有效性检查
    有效标准：
    1. 停止线点数量足够
    2. 停止线点在车辆左右两侧都有分布
    3. 最前面的停止线点前方约两米内的车道线点云很少
    
    参数:
        stop_line_points: 当前窗口内的停止线点云
        window_points: 当前窗口内的所有点云
        pose: 当前位姿
        full_points: 整个点云数据，用于检查停止线前方更远处的点云
    """
    if len(stop_line_points) < MIN_STOP_LINE_POINTS:
        print(f"停止线点数量不足: {len(stop_line_points)} < {MIN_STOP_LINE_POINTS}")
        return False, None
    
    # 获取车辆位置和方向
    vehicle_pos = np.array(pose[:3], dtype=float)
    quat = np.array(pose[3:7], dtype=float)
    R = Rotation.from_quat(quat)
    forward = R.apply([1, 0, 0])  # 车辆前进方向
    left = R.apply([0, 1, 0])     # 车辆左侧方向
    
    # 检查停止线点云在车辆左右两侧的分布
    rel_points = stop_line_points[:, :3] - vehicle_pos
    left_proj = np.dot(rel_points, left)
    
    # 计算左右两侧的点数
    left_side_count = np.sum(left_proj < -0.5)  # 车辆右侧（负值）
    right_side_count = np.sum(left_proj > 0.5)  # 车辆左侧（正值）
    
    # 确保停止线在左右两侧都有足够的点
    if left_side_count < MIN_SIDE_POINTS or right_side_count < MIN_SIDE_POINTS:
        print(f"停止线分布不均匀: 左侧 {right_side_count} 点, 右侧 {left_side_count} 点")
        return False, None
    
    print(f"停止线分布: 左侧 {right_side_count} 点, 右侧 {left_side_count} 点")

    # ==================== 新增：几何形状验证 ====================
    if len(stop_line_points) > 10:
        points_for_pca = stop_line_points[:, :3]
        cov_matrix = np.cov(points_for_pca, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 验证线性度：最大特征值应远大于第二大特征值
        linearity_ratio = eigenvalues[2] / (eigenvalues[1] + 1e-8)
        if linearity_ratio < 10:
            print(f"停止线形状非线性 (线性度比: {linearity_ratio:.2f} < 10)，判断为无效")
            return False, None

        # 验证方向：停止线方向应与车辆前进方向大致垂直
        stop_line_direction = eigenvectors[:, 2]
        dot_product = np.abs(np.dot(stop_line_direction, forward))
        if dot_product > 0.3: # cos(72.5 deg) ≈ 0.3. 允许约17.5度的偏差
            print(f"停止线方向与前进方向不垂直 (点积: {dot_product:.2f} > 0.3)，判断为无效")
            return False, None
        
        print(f"停止线几何形状验证通过 (线性度: {linearity_ratio:.2f}, 方向点积: {dot_product:.2f})")
    # ===========================================================
    
    # 计算停止线点云在前进方向上的投影
    forward_proj_stopline = np.dot(rel_points, forward)
    
    # 找到最前面的停止线点
    if len(forward_proj_stopline) == 0:
        return False, None
    
    # 直接找到最前面的停止线点（前进方向投影最大的点）
    front_idx = np.argmax(forward_proj_stopline)
    front_stopline_pos = stop_line_points[front_idx, :3]
    
    # 检查前方区域 - 使用完整点云数据
    FRONT_CHECK_EXTENDED_DIST = 8.0  # 扩展检查距离为8米
    
    if full_points is not None:
        # 从完整点云中提取停止线前方区域内的点
        rel_all_points = full_points[:, :3] - front_stopline_pos
        forward_proj = np.dot(rel_all_points, forward)
        left_proj = np.dot(rel_all_points, left)
        
        # 提取停止线前方区域内的点 (0-8米)，并限制左右范围（各15米）
        front_mask_near = (forward_proj > 0) & (forward_proj < FRONT_CHECK_DIST) & \
                         (np.abs(left_proj) < 15.0)  # 左右各限制15米
        front_points_near = full_points[front_mask_near]
        
        # 计算前方区域内的车道线点数
        front_lane_points_near = filter_points_by_class(front_points_near, [LANE_SOLID_CLASS, LANE_DASHED_CLASS])
        front_stop_points_near = filter_points_by_class(front_points_near, [STOP_LINE_CLASS])
        
        front_lane_count_near = len(front_lane_points_near)
        front_stop_count_near = len(front_stop_points_near)
        
        print(f"停止线前方 {FRONT_CHECK_DIST}m 内（左右各15m）的车道线点数: {front_lane_count_near}")
        print(f"停止线前方 {FRONT_CHECK_DIST}m 内（左右各15m）的停止线点数: {front_stop_count_near}")
        
        # 如果前方2米内车道线点数很少，并且8米范围内没有停止线点云，说明可能是路口
        if front_lane_count_near < MIN_LANE_POINTS and front_stop_count_near == 0:
            return True, front_stopline_pos
        else:
            if front_lane_count_near >= MIN_LANE_POINTS:
                print(f"前方车道线点数过多: {front_lane_count_near} >= {MIN_LANE_POINTS}")
            return False, None
    else:
        # 如果没有提供完整点云数据，则使用窗口点云（旧方法）
        rel_all_points = window_points[:, :3] - front_stopline_pos
        forward_proj = np.dot(rel_all_points, forward)
        
        front_mask = (forward_proj > 0) & (forward_proj < FRONT_CHECK_DIST)
        front_points = window_points[front_mask]
        
        front_lane_points = filter_points_by_class(front_points, [LANE_SOLID_CLASS, LANE_DASHED_CLASS])
        front_stop_points = filter_points_by_class(front_points, [STOP_LINE_CLASS])
        front_lane_count = len(front_lane_points)
        front_stop_count = len(front_stop_points)
        
        print(f"停止线前方 {FRONT_CHECK_DIST}m 内的车道线点数: {front_lane_count}")
        print(f"停止线前方 {FRONT_CHECK_DIST}m 内的停止线点数: {front_stop_count}")
        
        if front_lane_count < MIN_LANE_POINTS and front_stop_count == 0:
            return True, front_stopline_pos
        else:
            print(f"前方车道线点数过多: {front_lane_count} >= {MIN_LANE_POINTS}")
            return False, None

def visualize_front_lane_points(front_lane_points, stop_line_points, center_pos, forward, left):
    pass # 可视化函数省略

def detect_lane_start(window_points, pose, min_lane_points=MIN_LANE_POINTS):
    """
    检测是否进入车道（车道线点数突然增多）
    """
    lane_points = filter_points_by_class(window_points, [LANE_SOLID_CLASS, LANE_DASHED_CLASS])
    return len(lane_points) >= min_lane_points

def main(argv=None):
    parser = argparse.ArgumentParser(description="保存停止线点云数据")
    parser.add_argument("-i", "--input_file", type=str, help="输入点云文件路径", required=True)
    parser.add_argument("--pose_file", type=str, help="车辆位姿文件路径 (TUM格式)", required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="是否可视化结果", default=False)
    parser.add_argument("--visualize_front", action="store_true", help="是否可视化前方车道线点云", default=False)
    parser.add_argument("--min_points", type=int, default=MIN_STOP_LINE_POINTS, help="最小停止线点数")
    parser.add_argument("--window_length", type=float, default=20.0, help="窗口长度")
    parser.add_argument("--window_width", type=float, default=20.0, help="窗口宽度")
    args = parser.parse_args(argv)
    
    print(f"加载点云数据: {args.input_file}")
    points_with_class = load_point_cloud_with_class(args.input_file)
    if points_with_class is None:
        return
    
    print("过滤点云数据...")
    needed_classes = [LANE_SOLID_CLASS, LANE_DASHED_CLASS, LANE_ARROW_CLASS, STOP_LINE_CLASS]
    filtered_points = filter_points_by_class(points_with_class, needed_classes)
    print(f"过滤后点云数量: {len(filtered_points)}")
    
    print(f"加载位姿数据: {args.pose_file}")
    timestamps, poses = load_poses_tum(args.pose_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 状态变量
    segment_count = 0
    in_lane_segment = False
    current_segment_points = []
    current_segment_poses = []
    segment_start_idx = -1
    
    # 延迟截断相关变量
    truncation_pending = False
    target_stop_position = None

    # 记录已保存的停止线位置，避免重复保存
    saved_stopline_positions = []
    min_position_distance = 5.0
    
    saved_point_coords = set()
    coord_precision = 0.01
    
    for idx, (timestamp, pose) in enumerate(zip(timestamps, poses)):
        print(f"处理位姿 {idx+1}/{len(poses)}")
        
        # 提取当前窗口内的点云
        window_points = extract_window_by_pose(filtered_points, pose, 
                                              window_length=args.window_length, 
                                              window_width=args.window_width)
        
        # 只有在车道段内且没有等待截断时，才收集点云
        if in_lane_segment and not truncation_pending:
            lane_points = filter_points_by_class(window_points, [LANE_SOLID_CLASS, LANE_DASHED_CLASS, LANE_ARROW_CLASS, STOP_LINE_CLASS])
            new_points = []
            for point in lane_points:
                coord_key = tuple(np.round(point[:3] / coord_precision) * coord_precision)
                if coord_key not in saved_point_coords:
                    new_points.append(point)
                    saved_point_coords.add(coord_key)
            if len(new_points) > 0:
                current_segment_points.append(np.array(new_points))

        # 只要在车道段内，就持续收集位姿
        if in_lane_segment:
            current_segment_poses.append((idx, timestamp, pose))

        # 检查是否越过了目标停止线
        if truncation_pending:
            current_pos = np.array(pose[:3])
            forward_dir = get_direction_from_pose(pose)
            vec_to_stopline = target_stop_position - current_pos
            # 当到停止线的向量与前进方向的点积为负时，说明已越过
            if np.dot(vec_to_stopline, forward_dir) < 0:
                print(f"位姿 {idx+1}: 车辆已越过目标停止线，执行截断")
                # --- 执行保存逻辑 ---
                segment_count += 1
                if len(current_segment_points) > 0:
                    all_points = np.vstack(current_segment_points)
                    segment_filename = os.path.join(args.output_dir, f"lane_segment_{segment_count:03d}.pcd")
                    colors = []
                    for c in all_points[:, 3]:
                        if c == LANE_SOLID_CLASS: colors.append([0, 1, 0])
                        elif c == LANE_DASHED_CLASS: colors.append([1, 0, 0])
                        elif c == LANE_ARROW_CLASS: colors.append([1, 1, 0])
                        else: colors.append([0.5, 0.5, 0.5])
                    save_point_cloud(all_points, segment_filename, colors=np.array(colors))
                    
                    pose_filename = os.path.join(args.output_dir, f"lane_segment_{segment_count:03d}_poses.txt")
                    with open(pose_filename, 'w') as f:
                        f.write("# timestamp x y z qx qy qz qw\n")
                        for p_idx, p_timestamp, p_data in current_segment_poses:
                            f.write(f"{p_timestamp:.6f} {p_data[0]:.6f} {p_data[1]:.6f} {p_data[2]:.6f} "
                                   f"{p_data[3]:.6f} {p_data[4]:.6f} {p_data[5]:.6f} {p_data[6]:.6f}\n")
                else:
                    print(f"段 {segment_count}: 没有新的点云，跳过保存")
                
                # --- 重置状态 ---
                in_lane_segment = False
                truncation_pending = False
                target_stop_position = None
                current_segment_points = []
                current_segment_poses = []
                segment_start_idx = -1
                continue # 完成截断，进入下一次循环

        # 状态转换逻辑
        if not in_lane_segment:
            if detect_lane_start(window_points, pose):
                print(f"位姿 {idx+1}: 检测到车道起点")
                in_lane_segment = True
                # 开始收集点云和位姿
                lane_points = filter_points_by_class(window_points, [LANE_SOLID_CLASS, LANE_DASHED_CLASS, LANE_ARROW_CLASS, STOP_LINE_CLASS])
                new_points = []
                for point in lane_points:
                    coord_key = tuple(np.round(point[:3] / coord_precision) * coord_precision)
                    if coord_key not in saved_point_coords:
                        new_points.append(point)
                        saved_point_coords.add(coord_key)
                if len(new_points) > 0:
                    current_segment_points = [np.array(new_points)]
                else:
                    current_segment_points = []
                current_segment_poses = [(idx, timestamp, pose)]
                segment_start_idx = idx
        
        # 只有在车道内且没有等待截断时，才检测停止线
        if in_lane_segment and not truncation_pending:
            stop_line_points = filter_points_by_class(window_points, [STOP_LINE_CLASS])
            if len(stop_line_points) > 0:
                has_valid_stop_line, front_stopline_pos = is_valid_stop_line_simple(
                    stop_line_points, window_points, pose, full_points=filtered_points
                )
                
                if has_valid_stop_line:
                    is_duplicate = False
                    for saved_pos in saved_stopline_positions:
                        if np.linalg.norm(front_stopline_pos - saved_pos) < min_position_distance:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        print(f"位姿 {idx+1}: 检测到有效停止线，启动延迟截断模式")
                        truncation_pending = True
                        target_stop_position = front_stopline_pos
                        saved_stopline_positions.append(front_stopline_pos)

    # 处理最后一个未完成的段
    if in_lane_segment and len(current_segment_points) > 0:
        segment_count += 1
        print(f"警告：轨迹结束，保存未完成的段 {segment_count}")
        all_points = np.vstack(current_segment_points)
        segment_filename = os.path.join(args.output_dir, f"lane_segment_incomplete_{segment_count:03d}.pcd")
        colors = []
        for c in all_points[:, 3]:
            if c == LANE_SOLID_CLASS: colors.append([0, 1, 0])
            elif c == LANE_DASHED_CLASS: colors.append([1, 0, 0])
            elif c == LANE_ARROW_CLASS: colors.append([1, 1, 0])
            else: colors.append([0.5, 0.5, 0.5])
        save_point_cloud(all_points, segment_filename, colors=np.array(colors))
        # 保存位姿
        pose_filename = os.path.join(args.output_dir, f"lane_segment_incomplete_{segment_count:03d}_poses.txt")
        with open(pose_filename, 'w') as f:
            f.write("# timestamp x y z qx qy qz qw\n")
            for p_idx, p_timestamp, p_data in current_segment_poses:
                f.write(f"{p_timestamp:.6f} {p_data[0]:.6f} {p_data[1]:.6f} {p_data[2]:.6f} "
                       f"{p_data[3]:.6f} {p_data[4]:.6f} {p_data[5]:.6f} {p_data[6]:.6f}\n")
    
    print(f"处理完成。保存了 {segment_count} 个车道段。")

if __name__ == "__main__":
    main()
