#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d
import argparse
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import LineString, Point
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import math
import sys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline

# 定义类别ID
LANE_SOLID_CLASS = 1
LANE_DASHED_CLASS = 2
LANE_ARROW_CLASS = 3
STOP_LINE_CLASS = 5

def load_point_cloud(filepath):
    """加载点云文件"""
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # 根据颜色判断类别
        class_ids = np.zeros(len(points), dtype=int)
        for i, color in enumerate(colors):
            if np.allclose(color, [0, 1, 0]):  # 绿色 - 实线
                class_ids[i] = LANE_SOLID_CLASS
            elif np.allclose(color, [1, 0, 0]):  # 红色 - 虚线
                class_ids[i] = LANE_DASHED_CLASS
            elif np.allclose(color, [1, 1, 0]):  # 黄色 - 箭头
                class_ids[i] = LANE_ARROW_CLASS
            elif np.allclose(color, [0, 0, 1]):  # 蓝色 - 停止线
                class_ids[i] = STOP_LINE_CLASS
        
        # 合并点云坐标和类别
        points_with_class = np.column_stack((points, class_ids))
        return points_with_class
    except Exception as e:
        print(f"加载点云文件失败: {e}")
        return None

def load_poses(pose_file):
    """加载位姿文件 (TUM格式)"""
    try:
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                data = line.strip().split()
                if len(data) == 8:
                    timestamp = float(data[0])
                    x, y, z = float(data[1]), float(data[2]), float(data[3])
                    qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
                    poses.append((timestamp, [x, y, z, qx, qy, qz, qw]))
        return poses
    except Exception as e:
        print(f"加载位姿文件失败: {e}")
        return []

def calculate_pose_displacement(poses):
    """计算位姿序列的总位移"""
    if len(poses) < 2:
        return 0.0
    
    total_displacement = 0.0
    prev_pos = np.array(poses[0][1][:3])
    
    for _, pose in poses[1:]:
        curr_pos = np.array(pose[:3])
        displacement = np.linalg.norm(curr_pos - prev_pos)
        total_displacement += displacement
        prev_pos = curr_pos
    
    return total_displacement

def filter_points_by_class(points, class_ids):
    """根据类别ID过滤点云"""
    if len(points) == 0:
        return np.array([])
    mask = np.isin(points[:, 3], class_ids)
    return points[mask]

def flatten_lane_line_points(line):
    """将拟合出的车道线结构整理成 Nx3 点序列。"""
    arr = np.asarray(line, dtype=object)
    try:
        arr = np.asarray(line, dtype=float)
    except (TypeError, ValueError):
        flat_points = []
        for item in line:
            sub = flatten_lane_line_points(item)
            if sub is not None and len(sub) > 0:
                flat_points.extend(sub.tolist())
        return np.asarray(flat_points, dtype=float) if flat_points else np.empty((0, 3), dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] >= 2:
            return arr.reshape(1, -1)
        return np.empty((0, 3), dtype=float)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :3] if arr.shape[1] >= 3 else arr

def classify_lane_type_by_semantic_ratio(
    line,
    solid_lane_points,
    dashed_lane_points,
    solid_threshold=0.8,
    search_radius=0.6,
    fallback_type="dashed"
):
    """沿拟合车道线统计语义点，实线占比超过阈值才认定为实线。"""
    line_points = flatten_lane_line_points(line)
    if line_points is None or len(line_points) < 2:
        return fallback_type

    semantic_parts = []
    if solid_lane_points is not None and len(solid_lane_points) > 0:
        semantic_parts.append(solid_lane_points[:, :4])
    if dashed_lane_points is not None and len(dashed_lane_points) > 0:
        semantic_parts.append(dashed_lane_points[:, :4])
    if not semantic_parts:
        return fallback_type

    semantic_points = np.vstack(semantic_parts)
    line_xy = np.asarray(line_points[:, :2], dtype=float)
    point_xy = np.asarray(semantic_points[:, :2], dtype=float)
    min_dist = np.full(len(point_xy), np.inf)

    for start, end in zip(line_xy[:-1], line_xy[1:]):
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-8:
            continue
        rel = point_xy - start
        t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
        proj = start + t[:, None] * seg
        dist = np.linalg.norm(point_xy - proj, axis=1)
        min_dist = np.minimum(min_dist, dist)

    nearby = semantic_points[min_dist <= search_radius]
    if len(nearby) == 0:
        return fallback_type

    solid_count = int(np.count_nonzero(nearby[:, 3] == LANE_SOLID_CLASS))
    dashed_count = int(np.count_nonzero(nearby[:, 3] == LANE_DASHED_CLASS))
    total = solid_count + dashed_count
    if total == 0:
        return fallback_type

    solid_ratio = solid_count / total
    lane_type = "solid" if solid_ratio >= solid_threshold else "dashed"
    print(
        f"语义比例判定车道类型: solid={solid_count}, dashed={dashed_count}, "
        f"solid_ratio={solid_ratio:.3f}, type={lane_type}"
    )
    return lane_type

def cluster_lane_points(lane_points, eps=0.5, min_samples=10):
    """使用DBSCAN聚类车道线点云"""
    if len(lane_points) < min_samples:
        return []
    
    # 使用XYZ坐标进行聚类
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lane_points[:, :3])  # 使用3D坐标
    
    # 执行DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
    
    # 提取聚类结果
    clusters = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:  # 噪声点
            continue
        cluster_points = lane_points[labels == label]
        clusters.append(cluster_points)
    
    return clusters

def visualize_lane_segments_3d(lane_segments, stop_line_positions=None, title=None):
    """使用Open3D进行三维可视化车道线段"""
    # 创建可视化窗口
    window_title = title if title else "3D Lane Segments and Stop Lines"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1200, height=800)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # 添加车道线
    for i, segment_info in enumerate(lane_segments):
        # 检查segment_info的格式，可能是(点云, 类型)或(点云, 类型, 编号)
        # print(segment_info)
        if isinstance(segment_info, tuple):
            if len(segment_info) >= 2:
                segment = segment_info[0]
                lane_type = segment_info[1]
            else:
                segment = segment_info
                lane_type = "unknown"
        else:
            segment = segment_info
            lane_type = "unknown"
        
        # 确保点云有三维坐标
        if segment.shape[1] == 2:
            # 如果只有XY坐标，添加Z坐标（高度）
            z_coords = np.zeros((segment.shape[0], 1))
            segment_3d = np.hstack((segment, z_coords))
        else:
            segment_3d = segment[:, :3]
        
        # 创建线段
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(segment_3d)
        
        # 创建连接线
        lines = []
        for j in range(len(segment_3d) - 1):
            lines.append([j, j + 1])
        
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # 根据车道线类型设置颜色
        if lane_type == "solid":
            # 实线 - 绿色 [0, 1, 0]
            colors = np.array([[0, 1, 0] for _ in range(len(lines))])
        elif lane_type == "dashed":
            # 虚线 - 红色 [1, 0, 0]
            colors = np.array([[1, 0, 0] for _ in range(len(lines))])
        else:
            # 未知类型 - 灰色 [0.5, 0.5, 0.5]
            colors = np.array([[0.5, 0.5, 0.5] for _ in range(len(lines))])
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        vis.add_geometry(line_set)
    
    # 添加停止线
    if stop_line_positions is not None:
        for pos in stop_line_positions:
            # 创建停止线标记（球体）
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.translate(pos)
            sphere.paint_uniform_color([0, 0, 1])  # 蓝色
            vis.add_geometry(sphere)
    
    # 设置视角
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 3.0
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def visualize_lane_segments(lane_segments, stop_line_positions=None, title=None):
    """可视化车道线段（2D）"""
    plt.figure(figsize=(12, 10))
    
    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    
    for i, segment in enumerate(lane_segments):
        color = colors[i % len(colors)]
        plt.plot(segment[:, 0], segment[:, 1], color, linewidth=2, label=f'Lane {i+1}')
    
    if stop_line_positions:
        for i, pos in enumerate(stop_line_positions):
            plt.scatter(pos[0], pos[1], c='r', s=100, marker='x', label=f'Stop Line {i+1}' if i == 0 else '')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt_title = title if title else 'Lane Segments and Stop Lines (2D View)'
    plt.title(plt_title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

def project_line_to_trajectory(line_points, trajectory_direction, reference_point):
    """
    将拟合的线条投影到车辆轨迹方向上
    
    参数:
        line_points: 原始线条点
        trajectory_direction: 车辆轨迹方向向量
        reference_point: 参考点（车辆起始位置）
        
    返回:
        projected_line: 投影后的线条点
    """
    if len(line_points) < 2:
        return line_points
    
    # 计算线条点在轨迹方向上的投影
    projected_points = []
    for point in line_points:
        # 计算点到参考点的向量
        vec_to_point = np.array(point[:2]) - reference_point[:2]
        
        # 计算在轨迹方向上的投影距离
        projection_distance = np.dot(vec_to_point, trajectory_direction[:2])
        
        # 计算投影点位置
        projected_point = reference_point[:2] + projection_distance * trajectory_direction[:2]
        
        # 保持原始Z坐标
        if len(point) >= 3:
            projected_point = np.append(projected_point, point[2])
        else:
            projected_point = np.append(projected_point, 0)
        
        projected_points.append(projected_point)
    
    # 按投影距离排序
    projected_points = np.array(projected_points)
    distances = [np.dot(p[:2] - reference_point[:2], trajectory_direction[:2]) for p in projected_points]
    sorted_indices = np.argsort(distances)
    projected_points = projected_points[sorted_indices]
    
    return projected_points

def normalize_lane_line_length(lane_lines, reference_direction, dashed_length):
    """
    统一车道线的长度范围，使所有车道线的起始点和结尾点都在垂直于参考方向的平面上，
    并沿原始车道线方向进行拓展/裁剪以达到统一范围。
    
    参数:
        lane_lines: 车道线列表，每个元素为 (line_points, line_type, line_id)
        reference_direction: 参考方向向量（通常是虚线或车辆前进方向）
        target_length: 目标长度，如果为None则使用最长车道线的长度
    
    返回:
        normalized_lines: 长度统一且沿原始方向拓展的车道线列表
    """
    if not lane_lines:
        return lane_lines
    print('虚线长度: ', dashed_length)
    threshold = dashed_length/2
    # 计算每条车道线在参考方向上的投影范围
    direction_2d = reference_direction[:2]
    direction_2d = direction_2d / np.linalg.norm(direction_2d)
    
    line_ranges = []
    for line_points, line_type, line_id in lane_lines:
        if len(line_points) >= 2:
            line_2d = line_points[:, :2]
            projections = np.dot(line_2d, direction_2d)
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            line_ranges.append((min_proj, max_proj, line_points, line_type, line_id))
    
    if not line_ranges:
        return lane_lines
    
    # 确定统一的范围（使用最长车道线的范围）
    all_min = min([r[0] for r in line_ranges])
    all_max = max([r[1] for r in line_ranges])
    target_min = all_min
    target_max = all_max
    unified_length = target_max - target_min
    
    print(f"统一车道线范围: {target_min:.2f} 到 {target_max:.2f}")
    
    # 统一每条车道线的长度：沿原始车道线方向拓展/裁剪
    normalized_lines = []
    eps = 1e-6
    for min_proj, max_proj, line_points, line_type, line_id in line_ranges:
        if len(line_points) < 2:
            normalized_lines.append((line_points, line_type, line_id))
            continue
        
        # 原始车道线方向（2D），使用端点方向近似
        line_vec = line_points[-1] - line_points[0]
        line_dir_2d = line_vec[:2]
        norm_line = np.linalg.norm(line_dir_2d)
        if norm_line < eps:
            # 退化情形：保持原样
            normalized_lines.append((line_points, line_type, line_id))
            continue
        line_dir_2d = line_dir_2d / norm_line
        
        # 车道线中心点（2D）
        line_center_2d = (line_points[0][:2] + line_points[-1][:2]) / 2.0
        center_projection = np.dot(line_center_2d, direction_2d)
        
        # 计算与参考方向垂直平面（投影为 target_min/target_max）与原始车道方向直线的交点
        # 交点条件：dot(P0 + t*d_line, d_ref) = target_x
        dot_d = float(np.dot(line_dir_2d, direction_2d))
        if abs(dot_d) < eps:
            # 近似垂直：使用原逻辑（沿参考方向）回退，避免数值不稳定或无交
            start_point_2d = line_center_2d + direction_2d * (target_min - center_projection)
            end_point_2d = line_center_2d + direction_2d * (target_max - center_projection)
            
            # 选择性不延长：分别判断两端是否超过100m
            extend_start = max(0.0, min_proj - target_min)
            extend_end = max(0.0, target_max - max_proj)
            adj_min = target_min if extend_start <= threshold else min_proj
            adj_max = target_max if extend_end <= threshold else max_proj
            
            # 根据选择性延长后的范围重算端点
            start_point_2d = line_center_2d + direction_2d * (adj_min - center_projection)
            end_point_2d = line_center_2d + direction_2d * (adj_max - center_projection)
            
            z_avg = float(np.mean(line_points[:, 2]))
            new_start = np.array([start_point_2d[0], start_point_2d[1], z_avg])
            new_end = np.array([end_point_2d[0], end_point_2d[1], z_avg])
            new_points = [new_start]
            # 按与参考方向的投影在范围内筛选原始点
            proj_vals = np.dot(line_points[:, :2], direction_2d)
            keep_mask = (proj_vals >= min(adj_min, adj_max)) & (proj_vals <= max(adj_min, adj_max))
            kept = line_points[keep_mask]
            # 保持原顺序
            if kept.shape[0] > 0:
                new_points.extend(list(kept))
            new_points.append(new_end)
            new_line_points = np.vstack(new_points)
            normalized_lines.append((new_line_points, line_type, line_id))
            print(f"车道线 {line_id} ({line_type}): 近垂直回退，统一后范围 {target_min:.2f}-{target_max:.2f}")
            continue
        
        # 常规情形：按原始车道线方向求交点并重建
        t_start = (target_min - center_projection) / dot_d
        t_end = (target_max - center_projection) / dot_d
        
        # 确保起终次序
        t0, t1 = (t_start, t_end) if t_start <= t_end else (t_end, t_start)
        
        # 基于原始车道线方向参数 t 计算当前范围与目标范围的延长量，并选择性延长
        rel_ext = line_points[:, :2] - line_center_2d
        t_vals_ext = rel_ext @ line_dir_2d
        existing_t_min = float(np.min(t_vals_ext))
        existing_t_max = float(np.max(t_vals_ext))
        extend_start = max(0.0, existing_t_min - t0)
        extend_end = max(0.0, t1 - existing_t_max)
        adj_t0 = t0 if extend_start <= threshold else existing_t_min
        adj_t1 = t1 if extend_end <= threshold else existing_t_max
        
        start_point_2d = line_center_2d + adj_t0 * line_dir_2d
        end_point_2d = line_center_2d + adj_t1 * line_dir_2d
        
        # Z 使用原始平均高度
        z_avg = float(np.mean(line_points[:, 2]))
        new_start = np.array([start_point_2d[0], start_point_2d[1], z_avg])
        new_end = np.array([end_point_2d[0], end_point_2d[1], z_avg])
        
        # 以线方向参数 t 对原始点排序并截取 [t0, t1]
        rel = line_points[:, :2] - line_center_2d
        t_vals = rel @ line_dir_2d
        order = np.argsort(t_vals)
        t_sorted = t_vals[order]
        pts_sorted = line_points[order]
        mask = (t_sorted >= t0 - 1e-9) & (t_sorted <= t1 + 1e-9)
        pts_in_range = pts_sorted[mask]
        
        new_points = [new_start]
        if pts_in_range.shape[0] > 0:
            new_points.extend(list(pts_in_range))
        new_points.append(new_end)
        new_line_points = np.vstack(new_points)
        normalized_lines.append((new_line_points, line_type, line_id))
        
        print(f"车道线 {line_id} ({line_type}): 原始范围 {min_proj:.2f}-{max_proj:.2f}, 统一后范围 {target_min:.2f}-{target_max:.2f}")
    
    return normalized_lines

def fit_line_ransac(points, max_trials=1000, residual_threshold=0.2):
    """
    使用RANSAC拟合直线
    
    参数:
        points: 点云数组 (N, 3)
        max_trials: RANSAC最大迭代次数
        residual_threshold: 内点阈值
        
    返回:
        line_points: 拟合后的直线点 (2, 3)
        inliers: 内点索引
    """
    if len(points) < 2:
        return None, None
    
    # 提取XY坐标
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    
    # 使用RANSAC拟合直线
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        random_state=42
    )
    
    try:
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        
        # 计算直线的起点和终点
        x_min, x_max = np.min(X), np.max(X)
        x_range = np.array([[x_min], [x_max]])
        y_range = ransac.predict(x_range)
        
        # 创建直线点
        line_points = np.column_stack((x_range.flatten(), y_range.flatten()))
        
        # 添加Z坐标（使用内点的平均Z值）
        z_avg = np.mean(points[inlier_mask, 2])
        line_points = np.column_stack((line_points, np.array([z_avg, z_avg])))
        
        return line_points, inlier_mask
    except Exception as e:
        print(f"RANSAC直线拟合失败: {e}")
        return None, None


def fit_line_3d_ransac(points, max_trials=1000, residual_threshold=0.2):
    """
    使用RANSAC拟合三维直线
    
    参数:
        points: 点云数组 (N, 3)
        max_trials: RANSAC最大迭代次数
        residual_threshold: 内点阈值（米）
        
    返回:
        line_points: 拟合后的直线点 (2, 3)
        inliers: 内点索引
    """
    if len(points) < 2:
        return None, None
    
    # 方法1：分别拟合X-Y和X-Z平面，然后组合
    X = points[:, 0].reshape(-1, 1)
    y_xy = points[:, 1]  # Y坐标
    y_xz = points[:, 2]  # Z坐标
    
    try:
        # 拟合X-Y平面
        ransac_xy = RANSACRegressor(
            estimator=LinearRegression(),
            max_trials=max_trials,
            residual_threshold=residual_threshold,
            random_state=42
        )
        ransac_xy.fit(X, y_xy)
        
        # 拟合X-Z平面
        ransac_xz = RANSACRegressor(
            estimator=LinearRegression(),
            max_trials=max_trials,
            residual_threshold=residual_threshold,
            random_state=42
        )
        ransac_xz.fit(X, y_xz)
        
        # 计算直线的起点和终点
        x_min, x_max = np.min(X), np.max(X)
        x_range = np.array([[x_min], [x_max]])
        
        y_range = ransac_xy.predict(x_range)
        z_range = ransac_xz.predict(x_range)
        
        # 创建三维直线点
        line_points = np.column_stack((x_range.flatten(), y_range, z_range))
        
        # 计算内点（两个平面都满足条件的点）
        inlier_mask_xy = ransac_xy.inlier_mask_
        inlier_mask_xz = ransac_xz.inlier_mask_
        inlier_mask = inlier_mask_xy & inlier_mask_xz
        
        return line_points, inlier_mask
        
    except Exception as e:
        print(f"三维RANSAC直线拟合失败: {e}")
        return None, None

def fit_line_segments(points, max_gap=5.0, min_points=10, residual_threshold=0.2):
    """
    将点云拟合为多段直线
    
    参数:
        points: 点云数组 (N, 3)
        max_gap: 最大间隔距离，超过此距离视为不同线段
        min_points: 每段最少点数
        residual_threshold: RANSAC内点阈值
        
    返回:
        line_segments: 拟合后的直线段列表
    """
    if len(points) < min_points:
        return []
    
    # 按X坐标排序
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    # 检测点云中的间隙
    gaps = []
    for i in range(1, len(sorted_points)):
        dist = np.linalg.norm(sorted_points[i, :2] - sorted_points[i-1, :2])
        if dist > max_gap:
            gaps.append(i)
    
    # 分割点云
    segments = []
    start_idx = 0
    for gap_idx in gaps:
        segment = sorted_points[start_idx:gap_idx]
        if len(segment) >= min_points:
            segments.append(segment)
        start_idx = gap_idx
    
    # 添加最后一段
    if len(sorted_points[start_idx:]) >= min_points:
        segments.append(sorted_points[start_idx:])
    
    # 拟合每段点云
    line_segments = []
    for segment in segments:
        line_points, inliers = fit_line_ransac(segment, residual_threshold=residual_threshold)
        if line_points is not None:
            line_segments.append(line_points)
    
    return line_segments

def connect_lane_segments(lane_segments, max_distance=5.0, max_angle_diff=30.0):
    """
    连接相近且方向相似的车道线段
    
    参数:
        lane_segments: 车道线段列表，每个元素是(线段点, 类型)
        max_distance: 最大连接距离
        max_angle_diff: 最大角度差异（度）
        
    返回:
        connected_segments: 连接后的车道线段列表
    """
    if len(lane_segments) <= 1:
        return lane_segments
    
    # 计算每个线段的方向向量
    segment_directions = []
    for segment, _ in lane_segments:
        if len(segment) >= 2:
            direction = segment[-1, :2] - segment[0, :2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([0, 0])
            segment_directions.append(direction)
        else:
            segment_directions.append(np.array([0, 0]))
    
    # 计算每个线段的起点和终点
    segment_endpoints = []
    for segment, _ in lane_segments:
        if len(segment) >= 2:
            start_point = segment[0, :2]
            end_point = segment[-1, :2]
            segment_endpoints.append((start_point, end_point))
        else:
            # 单点无法形成线段，使用相同点作为起点和终点
            point = segment[0, :2]
            segment_endpoints.append((point, point))
    
    # 连接线段
    connected = [False] * len(lane_segments)
    connected_segments = []
    
    # 首先按类型分组
    solid_segments = []
    dashed_segments = []
    
    for i, (segment, lane_type) in enumerate(lane_segments):
        if lane_type == "solid":
            solid_segments.append((i, segment))
        else:
            dashed_segments.append((i, segment))
    
    # 处理实线
    connected_solid = connect_segments_by_type(solid_segments, segment_endpoints, segment_directions, 
                                              max_distance, max_angle_diff)
    
    # 处理虚线
    connected_dashed = connect_segments_by_type(dashed_segments, segment_endpoints, segment_directions, 
                                               max_distance, max_angle_diff)
    
    # 合并结果
    for segments, lane_type in [(connected_solid, "solid"), (connected_dashed, "dashed")]:
        for segment in segments:
            connected_segments.append((segment, lane_type))
    
    return connected_segments

def connect_segments_by_type(indexed_segments, segment_endpoints, segment_directions, max_distance, max_angle_diff):
    """
    连接同类型的线段
    
    参数:
        indexed_segments: 带索引的线段列表，每个元素是(索引, 线段点)
        segment_endpoints: 所有线段的端点列表
        segment_directions: 所有线段的方向向量列表
        max_distance: 最大连接距离
        max_angle_diff: 最大角度差异（度）
        
    返回:
        connected_segments: 连接后的线段列表
    """
    if len(indexed_segments) <= 1:
        return [segment for _, segment in indexed_segments]
    
    # 构建连接图
    connections = {}
    for i in range(len(indexed_segments)):
        connections[i] = []
    
    # 查找可能的连接
    for i in range(len(indexed_segments)):
        idx_i, segment_i = indexed_segments[i]
        start_i, end_i = segment_endpoints[idx_i]
        dir_i = segment_directions[idx_i]
        
        for j in range(i + 1, len(indexed_segments)):
            idx_j, segment_j = indexed_segments[j]
            start_j, end_j = segment_endpoints[idx_j]
            dir_j = segment_directions[idx_j]
            
            # 计算端点之间的距离
            dist_end_i_start_j = np.linalg.norm(end_i - start_j)
            dist_end_j_start_i = np.linalg.norm(end_j - start_i)
            
            # 计算方向向量之间的角度
            dot_product = np.clip(np.dot(dir_i, dir_j), -1.0, 1.0)
            angle_diff = np.degrees(np.arccos(np.abs(dot_product)))
            
            # 检查是否可以连接
            if dist_end_i_start_j < max_distance and angle_diff < max_angle_diff:
                connections[i].append((j, "end_i_start_j", dist_end_i_start_j))
            if dist_end_j_start_i < max_distance and angle_diff < max_angle_diff:
                connections[j].append((i, "end_j_start_i", dist_end_j_start_i))
    
    # 贪婪连接
    visited = set()
    connected_segments = []
    
    for i in range(len(indexed_segments)):
        if i in visited:
            continue
        
        # 开始一个新的连接链
        current_chain = [indexed_segments[i][1]]
        visited.add(i)
        
        # 向前连接
        current = i
        while True:
            # 查找最近的下一个线段
            next_segments = [(j, conn_type, dist) for j, conn_type, dist in connections[current] if j not in visited]
            if not next_segments:
                break
            
            # 选择最近的线段
            next_segment, conn_type, _ = min(next_segments, key=lambda x: x[2])
            visited.add(next_segment)
            
            # 添加到链中
            current_chain.append(indexed_segments[next_segment][1])
            current = next_segment
        
        # 连接线段
        if len(current_chain) > 1:
            # 合并线段点
            merged_segment = np.vstack(current_chain)
            connected_segments.append(merged_segment)
        else:
            # 单个线段，直接添加
            connected_segments.append(current_chain[0])
    
    # 添加未连接的线段
    for i in range(len(indexed_segments)):
        if i not in visited:
            connected_segments.append(indexed_segments[i][1])
    
    return connected_segments

def identify_and_number_lanes(lane_lines, pose):
    """
    对车道线进行编号，从右到左编号车道
    
    参数:
        lane_lines: 车道线列表，每个元素是(线段点, 类型)
        pose: 车辆位姿 [x, y, z, qx, qy, qz, qw]
        
    返回:
        numbered_lanes: 编号后的车道线列表，每个元素是(线段点, 类型, 编号)
        lane_count: 车道数量
    """
    if not lane_lines:
        return [], 0
    
    # 获取车辆位置和方向
    vehicle_pos = np.array(pose[:3], dtype=float)
    quat = np.array(pose[3:7], dtype=float)
    R = Rotation.from_quat(quat)
    forward = R.apply([1, 0, 0])  # 车辆前进方向
    left = R.apply([0, 1, 0])     # 车辆左侧方向
    
    # 计算每条车道线中点相对于车辆的横向位置
    lane_positions = []
    for i, (line, lane_type) in enumerate(lane_lines):
        # 计算线段中点
        midpoint = np.mean(line[:, :3], axis=0)
        
        # 计算中点相对于车辆的向量
        rel_vec = midpoint - vehicle_pos
        
        # 计算横向位置（左为正，右为负）
        lateral_pos = np.dot(rel_vec, left)
        
        lane_positions.append((i, lateral_pos, lane_type))
    
    # 按横向位置排序（从右到左）
    lane_positions.sort(key=lambda x: x[1])
    
    # 编号车道线
    numbered_lanes = []
    for lane_num, (lane_idx, _, _) in enumerate(lane_positions):
        line, lane_type = lane_lines[lane_idx]
        numbered_lanes.append((line, lane_type, lane_num))
    
    # 估计车道数量（车道线数量 + 1）
    lane_count = len(lane_positions) + 1
    
    return numbered_lanes, lane_count

from sklearn.linear_model import LinearRegression

def robust_fit_line(points, direction_vec, eps=0.5, min_samples=5, ransac_thresh=0.2):
    """
    先用一维DBSCAN去除孤立点，再RANSAC拟合，最后用内点最小二乘拟合
    points: (N, 3)
    direction_vec: 拟合方向向量（2D）
    eps: DBSCAN半径
    min_samples: DBSCAN最小点数
    ransac_thresh: RANSAC残差阈值
    返回: line_points(2,3) 或 None
    """
    if len(points) < min_samples:
        return None
    # 1. 一维投影
    projections = np.dot(points[:, :2], direction_vec[:2])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(projections.reshape(-1, 1))
    labels = db.labels_
    # 只保留主簇
    label_vals, label_counts = np.unique(labels[labels != -1], return_counts=True)
    if len(label_vals) == 0:
        return None
    main_label = label_vals[np.argmax(label_counts)]
    inlier_points = points[labels == main_label]
    if len(inlier_points) < 2:
        return None
    # 2. RANSAC拟合
    line, inliers = fit_line_ransac(inlier_points, residual_threshold=ransac_thresh)
    if line is None or inliers is None:
        return None
    # 3. 用RANSAC内点再做一次最小二乘拟合
    final_points = inlier_points[inliers]
    if len(final_points) < 2:
        return None
    X = final_points[:, 0].reshape(-1, 1)
    y = final_points[:, 1]
    model = LinearRegression().fit(X, y)
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = model.predict([[x_min]]), model.predict([[x_max]])
    z_avg = np.mean(final_points[:, 2])
    return np.array([[x_min, y_min[0], z_avg], [x_max, y_max[0], z_avg]])

def line_to_line_lateral_distance(line1, line2):
        """计算两条直线之间的横向距离"""
        p1, p2 = line1[0][:2], line1[1][:2]  # 第一条线
        p3, p4 = line2[0][:2], line2[1][:2]  # 第二条线
        
        # 计算两条线的方向向量
        v1 = p2 - p1
        v2 = p4 - p3
        
        # 归一化方向向量
        if np.linalg.norm(v1) > 1e-6:
            v1_norm = v1 / np.linalg.norm(v1)
        else:
            return np.linalg.norm(p1 - p3)
        
        if np.linalg.norm(v2) > 1e-6:
            v2_norm = v2 / np.linalg.norm(v2)
        else:
            return np.linalg.norm(p1 - p3)
        
        # 计算第一条线的法向量（垂直于线段方向）
        normal1 = np.array([-v1_norm[1], v1_norm[0]])
        
        # 计算第二条线的法向量
        normal2 = np.array([-v2_norm[1], v2_norm[0]])
        
        # 计算两条线中点之间的向量
        mid1 = (p1 + p2) / 2
        mid2 = (p3 + p4) / 2
        mid_vector = mid2 - mid1
        
        # 计算横向距离：中点向量在法向量上的投影
        lateral_dist1 = np.abs(np.dot(mid_vector, normal1))  # 相对于第一条线的横向距离
        lateral_dist2 = np.abs(np.dot(mid_vector, normal2))  # 相对于第二条线的横向距离
        
        # 返回较小的横向距离
        return min(lateral_dist1, lateral_dist2)

def calculate_line_quality(line_points):
    """
    计算线段质量，用于选择最佳的车道线
    
    Args:
        line_points: 线段点坐标 (N, 3)
    
    Returns:
        quality: 质量分数（越高越好）
    """
    if len(line_points) < 2:
        return 0.0
    
    # 计算线段长度
    line_length = np.linalg.norm(line_points[-1] - line_points[0])
    
    # 计算线段的直线度（点到直线的平均距离）
    if len(line_points) > 2:
        # 拟合直线
        line_dir = line_points[-1] - line_points[0]
        line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-8)
        
        # 计算所有点到直线的距离
        distances = []
        for point in line_points[1:-1]:  # 跳过端点
            # 点到直线的距离
            point_to_start = point - line_points[0]
            projection = np.dot(point_to_start, line_dir)
            projected_point = line_points[0] + projection * line_dir
            distance = np.linalg.norm(point - projected_point)
            distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            # 直线度 = 1 / (1 + 平均距离)
            straightness = 1.0 / (1.0 + avg_distance)
        else:
            straightness = 1.0
    else:
        straightness = 1.0
    
    # 综合质量分数：长度 * 直线度
    quality = line_length * straightness
    
    return quality

def filter_lane_lines_by_distance(lane_lines, min_distance=3.2, max_distance=4.5):
    """
    根据两两之间的横向距离筛选出合适的车道线对
    一般来说正确的两条车道线之间的距离应该在3.2米到4.5米之间
    
    Args:
        lane_lines: 车道线列表
        min_distance: 最小距离阈值（米）
        max_distance: 最大距离阈值（米）
    
    Returns:
        filtered_lines: 筛选后的车道线列表
    """
    if len(lane_lines) < 2:
        return lane_lines
    
    # 计算所有两两组合之间的距离
    line_pairs = []
    for i in range(len(lane_lines)):
        for j in range(i + 1, len(lane_lines)):
            distance = line_to_line_lateral_distance(lane_lines[i], lane_lines[j])
            line_pairs.append((i, j, distance))
    
    # 筛选出距离在合理范围内的线对
    valid_pairs = []
    for i, j, distance in line_pairs:
        if min_distance <= distance <= max_distance:
            valid_pairs.append((i, j, distance))
    
    if not valid_pairs:
        # 如果没有找到合适的线对，返回原始车道线
        return lane_lines
    
    # 统计每条线出现在有效线对中的次数
    line_counts = {}
    for i, j, distance in valid_pairs:
        line_counts[i] = line_counts.get(i, 0) + 1
        line_counts[j] = line_counts.get(j, 0) + 1
    
    # 选择出现次数最多的两条线
    if len(line_counts) >= 2:
        sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [sorted_lines[0][0], sorted_lines[1][0]]
        filtered_lines = [lane_lines[i] for i in selected_indices]
        return filtered_lines
    else:
        # 如果只有一条线，返回原始车道线
        return lane_lines

def fit_lane_segments_adaptive(points, direction_vec, dashed_length=None, block_size=20.0, min_points=8):
    """
    分块+自适应回退整体拟合（沿整体拟合直线段分块）
    points: (N, 3)
    direction_vec: 拟合主方向（2D）
    block_size: 分块长度（米）
    min_points: 每块最小点数
    返回：所有分块线段的列表
    """
    if len(points) < 2:
        return []
    # 整体拟合，得到一条直线段
    # global_line = fit_line_ransac(points)[0]
    
    global_line = robust_fit_line(points, direction_vec, eps=0.5, min_samples=50, ransac_thresh=0.2)
    global_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(global_line, direction_vec, angle_threshold=45.0)
    if global_line is None:
        return []
    # 计算整体直线段的起止点
    p0, p1 = global_line[0][:2], global_line[1][:2]
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-3:
        return []
    line_dir = line_vec / line_len
    # 沿整体直线段分块
    num_blocks = int(np.ceil(line_len / block_size))
    segments = []
    fallback_accum = 0.0
    pending_fallbacks = []
    if dashed_length is None:
        threshold = 100
    else:
        threshold = dashed_length/2 if dashed_length/2>150 else 150
    print('threshold: ', threshold)
    for i in range(num_blocks):
        seg_start = p0 + line_dir * (i * block_size)
        seg_end = p0 + line_dir * (min((i + 1) * block_size, line_len))
        block_length = min((i + 1) * block_size, line_len) - i * block_size
        # 选取投影到该分块区间内的点
        rel_vecs = points[:, :2] - p0
        proj_lens = np.dot(rel_vecs, line_dir)
        # 同时考虑左右±1.5米范围内的点（相对global_line的法向距离）
        normal_vec = np.array([-line_dir[1], line_dir[0]])
        lateral_offsets = np.dot(rel_vecs, normal_vec)
        lateral_mask = np.abs(lateral_offsets) <= 1.5
        mask = ((proj_lens >= i * block_size) & (proj_lens < min((i + 1) * block_size, line_len)) & lateral_mask)
        seg_points = points[mask]
        is_fallback = True
        line = None
        if len(seg_points) >= min_points:
            # 局部拟合
            line = fit_line_ransac_local(seg_points)
            line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(line, line_dir, angle_threshold=45.0)
            # print(f"第{i}个分块， 分块点数{len(seg_points)}")
            segment_lateral_distance = line_to_line_lateral_distance(line, global_line)
            print(f"第{i}个分块， 分块点数{len(seg_points)}， 分块距离{segment_lateral_distance}")
            if line is not None and segment_lateral_distance < 1.5:
                is_fallback = False
                # segments.append(line)
                # 可视化当前分块的点和拟合线段
                # try:
                # visualize_segment_3d_with_points(
                #     seg_points,
                #     [(np.array(line), "block")],
                #     None,
                #     segment_id=f"block_{i}"
                # )
                # except Exception as e:
                #     print(f"分块{i}可视化异常: {e}")
        #     else:
        #         # 回退到整体拟合
        #         segments.append(np.array([
        #             [seg_start[0], seg_start[1], np.mean(global_line[:, 2])],
        #             [seg_end[0], seg_end[1], np.mean(global_line[:, 2])]
        #         ]))
        # else:
        #     # 回退到整体拟合
        #     print(f"第{i}个分块， 分块点数{len(seg_points)}， 回退到整体拟合")
        #     segments.append(np.array([
        #         [seg_start[0], seg_start[1], np.mean(global_line[:, 2])],
        #         [seg_end[0], seg_end[1], np.mean(global_line[:, 2])]
        #     ]))
        if is_fallback:
            if len(seg_points) < min_points:
                print(f"第{i}个分块， 分块点数{len(seg_points)}， 回退到整体拟合")
            # 回退到整体拟合
            fallback_seg = np.array([
                [seg_start[0], seg_start[1], np.mean(global_line[:, 2])],
                [seg_end[0], seg_end[1], np.mean(global_line[:, 2])]
            ])
            pending_fallbacks.append(fallback_seg)
            fallback_accum += block_length
            if fallback_accum > threshold:
                pending_fallbacks = []
                fallback_accum = 0.0
        else:
            # Handle pending fallbacks
            if fallback_accum > 0:
                if fallback_accum <= threshold:
                    segments.extend(pending_fallbacks)
                pending_fallbacks = []
                fallback_accum = 0.0
            segments.append(line)

    # Handle any remaining pending fallbacks after the loop
    if fallback_accum > 0:
        if fallback_accum <= threshold:
            segments.extend(pending_fallbacks)

    return segments

def merge_segments_to_polyline(segments, direction_vec=None):
    
    """
    将多个拟合直线段按车辆方向融合为一条polyline
    
    Args:
        segments: list of (2,3) ndarray，拟合直线段
        vehicle_dir: 车辆方向向量 (2,) 或 (3,)
    
    Returns:
        polyline: (N,3) ndarray 或 None
    """
    if not segments:
        return None
    
    # 1. 计算每个线段起点在车辆方向上的投影
    vehicle_dir_2d = direction_vec[:2] / np.linalg.norm(direction_vec[:2])
    
    seg_info = []
    for seg in segments:
        if seg is None or len(seg) < 2:
            continue
            
        start_point = seg[0][:2]  # 取XY坐标
        end_point = seg[1][:2]

        middle_point = (start_point + end_point) / 2
        
        # 计算起点和终点在车辆方向上的投影
        start_proj = np.dot(start_point, vehicle_dir_2d)
        end_proj = np.dot(end_point, vehicle_dir_2d)
        middle_proj = np.dot(middle_point, vehicle_dir_2d)
        
        # 确保线段方向一致（起点在前，终点在后）
        # if end_proj < start_proj:
        #     seg = seg[::-1]  # 反转线段
        #     start_proj, end_proj = end_proj, start_proj
        
        seg_info.append({
            'segment': seg,
            'start_proj': start_proj,
            'end_proj': end_proj,
            'middle_proj': middle_proj,
            'start_point': seg[0],
            'end_point': seg[1]
        })
    
    if not seg_info:
        return None
    
    # 2. 按起点投影排序
    # seg_info.sort(key=lambda x: x['middle_proj'])
    
        # 3. 构建polyline
    polyline = []
    last_end_proj = float('-inf')
    
    for i, info in enumerate(segments):
        # 创建可视化窗口
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name="Polyline构建过程", width=1200, height=800)

        # current_start_proj = info['start_proj']
        # current_end_proj = info['end_proj']
        # current_seg = info['segment']
        current_seg = info
        
        # 检查是否存在交叉或顺序错乱
        # if current_start_proj <= last_end_proj + 1e-3:  # 允许小的数值误差
        #     # 存在交叉或重叠，跳过当前段
        #     continue
        # else:
            # 正常情况：当前段在上一段之后
        if not polyline:
            # 第一段，添加起点和终点
            polyline.append(current_seg[0])
            polyline.append(current_seg[1])
        else:
            # 后续段，只添加终点
            # polyline.append(current_seg[0])
            polyline.append(current_seg[1])
            
        # last_end_proj = current_end_proj
        
        # 可视化当前累积的polyline
        # print(f"步骤 {i+1}: 添加线段 {i}, polyline shape: {np.array(polyline).shape}")
        
        # # 清除之前的几何体
        # # vis.clear_geometries()
        
        # # 显示当前累积的polyline
        # # if len(polyline) >= 2:
        # polyline_array = np.array(polyline)
        
        # # 创建polyline的LineSet
        # line_set = o3d.geometry.LineSet()
        # points = o3d.utility.Vector3dVector(polyline_array)
        # lines = [[j, j+1] for j in range(len(polyline_array)-1)]
        # line_set.points = points
        # line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.paint_uniform_color([0, 1, 0])  # 绿色显示累积的polyline
        # vis.add_geometry(line_set)
        
        # # 显示polyline的起始端点（用不同颜色的圆形表示）
        # # 起始点（第一个点）- 蓝色圆形
        # start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # start_sphere.translate(polyline_array[0])
        # start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
        # vis.add_geometry(start_sphere)
        
        # # 终点（最后一个点）- 紫色圆形
        # end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # end_sphere.translate(polyline_array[-1])
        # end_sphere.paint_uniform_color([1, 0, 1])  # 紫色
        # vis.add_geometry(end_sphere)
        
        # # 显示当前正在处理的线段
        # current_line_set = o3d.geometry.LineSet()
        # current_points = o3d.utility.Vector3dVector(current_seg)
        # current_lines = [[0, 1]]
        # current_line_set.points = current_points
        # current_line_set.lines = o3d.utility.Vector2iVector(current_lines)
        # current_line_set.paint_uniform_color([1, 0, 0])  # 红色显示当前线段
        # vis.add_geometry(current_line_set)
        
        # # 显示所有原始线段（半透明）
        # for j, seg_info_item in enumerate(seg_info):
        #     seg = seg_info_item['segment']
        #     seg_line_set = o3d.geometry.LineSet()
        #     seg_points = o3d.utility.Vector3dVector(seg)
        #     seg_lines = [[0, 1]]
        #     seg_line_set.points = seg_points
        #     seg_line_set.lines = o3d.utility.Vector2iVector(seg_lines)
        #     if j == i:
        #         seg_line_set.paint_uniform_color([1, 0, 0])  # 当前线段红色
        #     else:
        #         seg_line_set.paint_uniform_color([0.5, 0.5, 0.5])  # 其他线段灰色
        #     vis.add_geometry(seg_line_set)
        
        # # 更新可视化
        # # vis.poll_events()
        # # vis.update_renderer()
        
        # # 关闭可视化窗口
        # vis.run()
        # # vis.destroy_window()
        # vis.destroy_window()
    
    
    
    if len(polyline) < 2:
        return None
    
    print("polyline shape: ", np.array(polyline).shape)
    return np.array(polyline)

def fit_multi_lanes(solid_points, center_line, normal_vec, solid_dashed_points=None, dashed_length=None, class_type=LANE_SOLID_CLASS, min_dist=2.5, max_dist=5.0, min_points=10, direction=1):
    """
    递归分离多条平行车道线
    """
    # --- 新增健壮性判断 ---
    if solid_points is None or len(solid_points) == 0 or center_line is None or len(center_line) == 0:
        return []
    results = []
    current_center = center_line
    while True:
        # 取中心线起点
        center_point = current_center[0][:2]
        distances = []
        for pt in solid_points:
            rel_vec = pt[:2] - center_point
            dist = np.dot(rel_vec, normal_vec)
            distances.append(dist)
        distances = np.array(distances)
        # 只保留在当前方向、2.0~5.0米范围内的点
        mask = (distances * direction > 0) & (np.abs(distances) >= min_dist) & (np.abs(distances) <= max_dist)
        candidate_points = solid_points[mask]
        print(f"实线点数={len(candidate_points)}")
        if len(candidate_points) < min_points:
            break
        
        if class_type == LANE_DASHED_CLASS:
            candidate_points = expand_fusion_points(candidate_points, solid_dashed_points, class_type)
        # 分块拟合
        segments = fit_lane_segments_adaptive(candidate_points, normal_vec, dashed_length, block_size=50.0, min_points=150)
        # 可视化分块线段和点云
        if len(segments) > 0 and len(candidate_points) > 0:
            try:
                visualize_segment_3d_with_points(candidate_points, [], None, segment_id=f"block_{len(results)}", block_segments=segments)
            except Exception as e:
                print(f"分块可视化异常: {e}")
        if not segments:
            break
        # 合并分块线段为一条车道线
        polyline = merge_segments_to_polyline(segments, normal_vec)
        # polyline_segments = []
        # for i in range(len(polyline) - 1):
        #     segment = np.array([polyline[i], polyline[i+1]])  # (2, 3)
        #     polyline_segments.append((segment, "merged"))
        # # 可视化融合后的polyline和对应点云
        # if polyline is not None and len(polyline) > 1:
        #     try:
        #         # 将polyline转换为线段序列                
        #         visualize_segment_3d_with_points(
        #             candidate_points,
        #             polyline_segments,
        #             None,
        #             segment_id=f"merged_{len(results)}"
        #         )
        #     except Exception as e:
        #         print(f"融合车道线可视化异常: {e}")
        # 计算平均距离
        mean_dist = np.mean([np.dot(pt[:2] - current_center[0][:2], normal_vec) for pt in polyline])
        results.append((polyline, mean_dist))
        current_center = polyline
    return results

def separate_solid_lanes_by_distribution(solid_points, vehicle_dir, min_lane_distance=2.5, max_lane_distance=4.5):
    """
    更鲁棒的实线车道分离，能处理噪声点云
    
    Args:
        solid_points: 实线点云 (N, 4)
        vehicle_dir: 车辆方向向量 (2,)
        min_lane_distance: 最小车道间距（更宽松）
        max_lane_distance: 最大车道间距（更宽松）
    
    Returns:
        lane_lines: 分离出的车道线列表
    """
    if len(solid_points) < 20:  # 增加最小点数要求
        return []
    
    coords_2d = solid_points[:, :2]
    
    # 1. 计算法向量
    normal_vec = np.array([-vehicle_dir[1], vehicle_dir[0]])
    
    # 2. 将点投影到法向量上
    proj = np.dot(coords_2d, normal_vec)
    
    # 3. 预处理：去除极端离群点
    proj_mean = np.median(proj)
    proj_std = np.std(proj)
    valid_mask = (proj > proj_mean - 3 * proj_std) & (proj < proj_mean + 3 * proj_std)
    filtered_points = solid_points[valid_mask]
    filtered_proj = proj[valid_mask]
    
    if len(filtered_points) < 10:
        return []
    
    # 4. 使用更鲁棒的聚类方法
    # 方法1：自适应DBSCAN
    from sklearn.cluster import DBSCAN
    
    # 自适应eps参数
    eps_adaptive = np.percentile(np.abs(np.diff(np.sort(filtered_proj))), 75) * 2
    eps_adaptive = max(0.5, min(2.0, eps_adaptive))  # 限制在合理范围内
    
    # 自适应min_samples
    min_samples_adaptive = max(80, min(10, len(filtered_points) // 20))
    
    db = DBSCAN(eps=eps_adaptive, min_samples=min_samples_adaptive).fit(filtered_proj.reshape(-1, 1))
    labels = db.labels_
    
    # 5. 统计每个聚类的中心位置
    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) < 2:
        return []
    
    lane_centers = []
    lane_points_list = []
    
    for label in unique_labels:
        mask = labels == label
        lane_points = filtered_points[mask]
        lane_center = np.median(filtered_proj[mask])
        lane_centers.append(lane_center)
        lane_points_list.append(lane_points)
    
    # 6. 按中心位置排序
    sorted_indices = np.argsort(lane_centers)
    lane_centers = [lane_centers[i] for i in sorted_indices]
    lane_points_list = [lane_points_list[i] for i in sorted_indices]
    
    print("车道中心个数: ", len(lane_centers))
    # 7. 更宽松的车道间距检查
    valid_lanes = []
    for i, (center, points) in enumerate(zip(lane_centers, lane_points_list)):
        if len(points) >= min_samples_adaptive:  # 使用自适应最小点数
            # 检查与相邻车道的间距（更宽松的条件）
            is_valid = True
            if i > 0:
                distance = abs(center - lane_centers[i-1])
                if distance < min_lane_distance or distance > max_lane_distance:
                    # 如果间距不合理，但点数足够多，仍然保留
                    # if len(points) >= min_samples_adaptive * 2:
                    #     is_valid = True
                    # else:
                    is_valid = False
            if i < len(lane_centers) - 1:
                distance = abs(center - lane_centers[i+1])
                if distance < min_lane_distance or distance > max_lane_distance:
                    # if len(points) >= min_samples_adaptive * 2:
                    #     is_valid = True
                    # else:
                    is_valid = False
            
            if is_valid:
                valid_lanes.append((center, points))
    
    # 8. 对每个有效车道进行拟合
    lane_lines = []
    for center, points in valid_lanes:
        if len(points) >= min_samples_adaptive:
            # 使用更鲁棒的拟合方法，确保方向与车辆方向一致
            try:
                line_result = fit_line_3d_ransac(points[:, :3])
                if line_result[0] is not None:
                    # 确保拟合的直线段方向与车辆方向一致
                    fitted_line = line_result[0]
                    # 检查并修正方向
                    corrected_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(
                        fitted_line, vehicle_dir, angle_threshold=45.0)
                    if direction_consistent:
                        lane_lines.append(corrected_line)
                    else:
                        # 如果方向不一致，使用备用方法
                        line_points, _ = fit_line_3d_ransac(points[:, :3])
                        if line_points is not None:
                            # 对备用方法的结果也进行方向修正
                            corrected_line, _, _ = check_and_correct_dashed_line_direction(
                                line_points, vehicle_dir, angle_threshold=45.0)
                            lane_lines.append(corrected_line)
            except:
                # 如果拟合失败，尝试PCA方法
                try:
                    line_points, _ = fit_line_3d_ransac(points[:, :3])
                    if line_points is not None:
                        # 对备用方法的结果也进行方向修正
                        corrected_line, _, _ = check_and_correct_dashed_line_direction(
                            line_points, vehicle_dir, angle_threshold=45.0)
                        lane_lines.append(corrected_line)
                except:
                    continue
    
    return lane_lines

def determine_lane_positions(lane_lines, vehicle_dir):
    """
    根据车道线相对于车辆方向的横向位置确定左右关系
    支持任意数量的车道线，特别优化了四条车道线的情况
    车辆左侧为左车道，车辆右侧为右车道
    
    Args:
        lane_lines: 车道线列表
        vehicle_dir: 车辆方向向量
    
    Returns:
        按左右顺序排列的车道线列表
    """
    if len(lane_lines) == 0:
        return []
    
    # 计算法向量（指向车辆左侧）
    normal_vec = np.array([-vehicle_dir[1], vehicle_dir[0]])
    
    # 计算每条车道线的中心位置
    lane_positions = []
    for i, line in enumerate(lane_lines):
        line_center = (line[0][:2] + line[1][:2]) / 2
        # 计算相对于车辆方向的横向位置
        lateral_pos = np.dot(line_center, normal_vec)
        lane_positions.append((lateral_pos, line, i))
    
    # 按横向位置排序（从车辆左侧到右侧）
    # lateral_pos > 0 表示在车辆左侧
    # lateral_pos < 0 表示在车辆右侧
    lane_positions.sort(key=lambda x: x[0], reverse=True)  # 从大到小，即从左到右
    
    # 打印车道线位置信息
    print(f"检测到 {len(lane_lines)} 条车道线:")
    for i, (lateral_pos, line, original_idx) in enumerate(lane_positions):
        position_name = ""
        if len(lane_positions) == 4:
            # 四条车道线的情况
            if i == 0:
                position_name = "最左侧车道"
            elif i == 1:
                position_name = "左侧车道"
            elif i == 2:
                position_name = "右侧车道"
            elif i == 3:
                position_name = "最右侧车道"
        elif len(lane_positions) == 3:
            # 三条车道线的情况
            if i == 0:
                position_name = "左侧车道"
            elif i == 1:
                position_name = "中间车道"
            elif i == 2:
                position_name = "右侧车道"
        elif len(lane_positions) == 2:
            # 两条车道线的情况
            if i == 0:
                position_name = "左侧车道"
            elif i == 1:
                position_name = "右侧车道"
        else:
            position_name = f"车道 {i+1}"
        
        print(f"  {position_name}: 横向位置 {lateral_pos:.2f}m (原始索引: {original_idx})")
    
    # 计算相邻车道线之间的距离
    if len(lane_positions) > 1:
        print("相邻车道线间距:")
        for i in range(len(lane_positions) - 1):
            current_pos = lane_positions[i][0]
            next_pos = lane_positions[i + 1][0]
            distance = abs(current_pos - next_pos)
            print(f"  车道 {i} 和车道 {i+1}: {distance:.2f}m")
    
    # 返回排序后的车道线
    return [pos[1] for pos in lane_positions]

# 验证直线段的连续性
def validate_segment_continuity(current_segments, previous_segments, continuity_threshold=0.5):
    """
    验证当前轮次拟合的直线段与上一轮次直线段的连续性
    
    Args:
        current_segments: 当前轮次的直线段列表
        previous_segments: 上一轮次的直线段列表
        continuity_threshold: 连续性阈值（米）
    
    Returns:
        valid_segments: 通过连续性验证的直线段列表
    """
    if not previous_segments:
        return current_segments
    
    valid_segments = []
    
    for current_seg in current_segments:
        if current_seg is None or len(current_seg) < 2:
            continue
        
        current_start = current_seg[0]
        is_continuous = False
        
        # 检查与上一轮次所有线段的连续性
        for prev_seg in previous_segments:
            if prev_seg is None or len(prev_seg) < 2:
                continue
            
            prev_end = prev_seg[1]
            
            # 计算当前起点与上一终点的距离
            distance = np.linalg.norm(current_start - prev_end)
            
            if distance <= continuity_threshold:
                is_continuous = True
                break
        
        if is_continuous:
            valid_segments.append(current_seg)
    
    return valid_segments

def sliding_window_lane_fitting(points, vehicle_forward_direction, dashed_class=2, solid_class=1, window_length=20.0, window_width=15.0, window_height=5.0):
    coords_3d = points[:, :3]
    vehicle_dir = vehicle_forward_direction[:2] / np.linalg.norm(vehicle_forward_direction[:2])
    proj = np.dot(coords_3d[:, :2], vehicle_dir)
    min_proj = np.percentile(proj, 1)
    max_proj = np.percentile(proj, 98)
    lane_segments = []
    dashed_segments = []
    left_solid_segments = []
    right_solid_segments = []

    # 记录上一个窗口的拟合方向
    previous_direction = vehicle_forward_direction
    
    # 窗口ID计数器
    window_id = 0

    for seg_start in np.arange(min_proj, max_proj, window_length):
        seg_end = seg_start + window_length
        current_window_segments = {}

        # 使用上一个窗口的方向（除了第一个窗口）
        if len(dashed_segments) > 0 or len(left_solid_segments) > 0 or len(right_solid_segments) > 0:
            # 使用上一个窗口的拟合方向
            current_vehicle_dir = previous_direction[:2] / np.linalg.norm(previous_direction[:2])
        else:
            # 第一个窗口使用原始车辆方向
            current_vehicle_dir = vehicle_dir

        # 沿车辆方向的投影筛选
        mask_length = (proj >= seg_start) & (proj < seg_end)
        
        # 横向投影筛选
        ortho_proj = np.dot(coords_3d[:, :2], np.array([-current_vehicle_dir[1], current_vehicle_dir[0]]))
        ortho_center = np.median(ortho_proj)
        mask_width = (ortho_proj >= ortho_center - window_width/2) & (ortho_proj <= ortho_center + window_width/2)
        
        # Z轴高度筛选
        z_center = np.median(coords_3d[mask_length & mask_width, 2]) if np.any(mask_length & mask_width) else 0
        mask_height = (coords_3d[:, 2] >= z_center - window_height/2) & (coords_3d[:, 2] <= z_center + window_height/2)
        
        # 三维窗口筛选
        mask = mask_length & mask_width & mask_height
        window_points = points[mask]
        if len(window_points) < 2:
            continue
        # 1. 拟合虚线
        dashed_points = window_points[window_points[:, 3] == dashed_class]
        if len(dashed_points) >= 80:
            # 对虚线点云进行分离拟合
            # lane_lines = separate_solid_lanes_by_distribution(dashed_points, current_vehicle_dir)
            lane_lines, cluster_points = separate_dashed_lanes_by_fitting(dashed_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=4.5)
            
            if len(lane_lines) == 1:
                print("只有一条虚线")
                # 只有一条虚线，按照原来的方式处理
                # line_result = fit_line_3d_ransac(dashed_points[:, :3])
                if lane_lines[0] is not None:
                    dashed_line = lane_lines[0]
                    dashed_line = clip_line_to_window(dashed_line, seg_start, seg_end, current_vehicle_dir)
                    if type(dashed_line) == np.ndarray:
                        dashed_segments.append([dashed_line[0].tolist(), dashed_line[1].tolist()])
                    else:
                        dashed_segments.append(dashed_line)
                    # 计算法向量
                    line_vec = dashed_line[1][:2] - dashed_line[0][:2]
                    line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-8)
                    normal_vec = np.array([-line_vec[1], line_vec[0]])
                    # 2. 区分左右实线
                    solid_points = window_points[window_points[:, 3] == solid_class]
                    if len(solid_points) >= 2:
                        # 投影到法向量
                        solid_proj = np.dot(solid_points[:, :2] - dashed_line[0][:2], normal_vec)
        
                        # 左实线：在虚线左侧2.0到4.5米范围内
                        left_mask = (solid_proj > 2.0) & (solid_proj <= 4.5)
                        left_points = solid_points[left_mask]
                        
                        # 右实线：在虚线右侧2.0到4.5米范围内
                        right_mask = (solid_proj < -2.0) & (solid_proj >= -4.5)
                        right_points = solid_points[right_mask]
                        
                        # 左实线拟合
                        if len(left_points) >= 2:
                            left_line = fit_line_3d_ransac(left_points[:, :3])
                            left_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(left_line[0], vehicle_dir, angle_threshold=45.0)
                            left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                            if left_line[0] is not None:
                                left_solid_segments.append(left_line)
                        
                        # 右实线拟合
                        if len(right_points) >= 2:
                            right_line = fit_line_3d_ransac(right_points[:, :3])
                            right_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(right_line[0], vehicle_dir, angle_threshold=45.0)
                            right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                            if right_line[0] is not None:
                                right_solid_segments.append(right_line)
                        
                        # 收集当前窗口的拟合结果
                        current_window_segments = {
                            'dashed': [dashed_line],
                            'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 and len(left_points) >= 2 else [],
                            'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 and len(right_points) >= 2 else []
                        }
            
            elif len(lane_lines) == 2:
                print("有两条虚线")
                # 有两条虚线，根据车辆方向分别处理
                # 对两条虚线分别拟合
                left_dashed_line = None
                right_dashed_line = None
                # 使用更稳健的separate_solid_lanes_by_distribution函数分离虚线
                dashed_lane_lines, _ = separate_dashed_lanes_by_fitting(dashed_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=4.5)
                
                # 根据分离结果确定左右虚线
                left_dashed_line = None
                right_dashed_line = None
                
                if len(dashed_lane_lines) >= 2:
                    # 计算每条线的中心点，确定左右位置
                    line_centers = []
                    for line in dashed_lane_lines:
                        if len(line) >= 2:
                            center = (line[0][:2] + line[1][:2]) / 2
                            line_centers.append((center, line))
                    
                    # 根据横向位置排序
                    normal_vec = np.array([-current_vehicle_dir[1], current_vehicle_dir[0]])
                    line_centers.sort(key=lambda x: np.dot(x[0], normal_vec), reverse=True)
                    
                    # 左侧虚线（横向位置最大）
                    if len(line_centers) >= 1:
                        left_dashed_line = line_centers[0][1]
                        
                        # 检查左侧虚线方向与车辆方向的一致性
                        if len(left_dashed_line) >= 2:
                            line_vec = left_dashed_line[1][:2] - left_dashed_line[0][:2]
                            line_length = np.linalg.norm(line_vec)
                            if line_length > 1e-8:
                                line_direction = line_vec / line_length
                                vehicle_dir_norm = current_vehicle_dir / (np.linalg.norm(current_vehicle_dir) + 1e-8)
                                
                                # 计算角度差
                                dot_product = np.clip(np.dot(line_direction, vehicle_dir_norm), -1.0, 1.0)
                                angle_diff = np.arccos(abs(dot_product)) * 180.0 / np.pi
                                
                                # 如果角度差超过45度，进行方向修正
                                if angle_diff > 45.0:
                                    print(f"左侧虚线方向与车辆方向不一致，角度差: {angle_diff:.2f}度，进行修正")
                                    
                                    # 将虚线投影到车辆方向
                                    center_point = (left_dashed_line[0] + left_dashed_line[1]) / 2
                                    extension_length = line_length / 2
                                    start_point = center_point[:2] - vehicle_dir_norm * extension_length
                                    end_point = center_point[:2] + vehicle_dir_norm * extension_length
                                    
                                    # 保持Z坐标不变
                                    z_start = left_dashed_line[0][2]
                                    z_end = left_dashed_line[1][2]
                                    
                                    left_dashed_line = np.array([
                                        [start_point[0], start_point[1], z_start],
                                        [end_point[0], end_point[1], z_end]
                                    ])
                                    print(f"已修正左侧虚线方向")
                        # left_dashed_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                        # dashed_segments.append(left_dashed_line)
                    
                    # 右侧虚线（横向位置最小）
                    if len(line_centers) >= 2:
                        right_dashed_line = line_centers[-1][1]
                        
                        # 检查右侧虚线方向与车辆方向的一致性
                        if len(right_dashed_line) >= 2:
                            line_vec = right_dashed_line[1][:2] - right_dashed_line[0][:2]
                            line_length = np.linalg.norm(line_vec)
                            if line_length > 1e-8:
                                line_direction = line_vec / line_length
                                vehicle_dir_norm = current_vehicle_dir / (np.linalg.norm(current_vehicle_dir) + 1e-8)
                                
                                # 计算角度差
                                dot_product = np.clip(np.dot(line_direction, vehicle_dir_norm), -1.0, 1.0)
                                angle_diff = np.arccos(abs(dot_product)) * 180.0 / np.pi
                                
                                # 如果角度差超过45度，进行方向修正
                                if angle_diff > 45.0:
                                    print(f"右侧虚线方向与车辆方向不一致，角度差: {angle_diff:.2f}度，进行修正")
                                    
                                    # 将虚线投影到车辆方向
                                    center_point = (right_dashed_line[0] + right_dashed_line[1]) / 2
                                    extension_length = line_length / 2
                                    start_point = center_point[:2] - vehicle_dir_norm * extension_length
                                    end_point = center_point[:2] + vehicle_dir_norm * extension_length
                                    
                                    # 保持Z坐标不变
                                    z_start = right_dashed_line[0][2]
                                    z_end = right_dashed_line[1][2]
                                    
                                    right_dashed_line = np.array([
                                        [start_point[0], start_point[1], z_start],
                                        [end_point[0], end_point[1], z_end]
                                    ])
                                    print(f"已修正右侧虚线方向")
                        # right_dashed_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)
                        # dashed_segments.append(right_dashed_line)
                
                # 处理实线
                solid_points = window_points[window_points[:, 3] == solid_class]
                dashed_line_list = []
                if len(solid_points) >= 2:
                    # 左侧实线：在左侧虚线左侧2.0~4.5米之间
                    if left_dashed_line is not None:
                        line_vec = left_dashed_line[1][:2] - left_dashed_line[0][:2]
                        line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-8)
                        normal_vec = np.array([-line_vec[1], line_vec[0]])
                        solid_proj = np.dot(solid_points[:, :2] - left_dashed_line[0][:2], normal_vec)
                        left_mask = (solid_proj > 2.0) & (solid_proj <= 4.5)
                        left_points = solid_points[left_mask]
                        
                        if len(left_points) >= 100:
                            left_line = fit_line_3d_ransac(left_points[:, :3])
                            left_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(left_line[0], vehicle_dir, angle_threshold=45.0)
                            left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                            if left_line[0] is not None:
                                left_solid_segments.append(left_line)
                                left_dashed_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                                # dashed_segments.append(left_dashed_line)
                                dashed_line_list.append(left_dashed_line)
                        else:
                            print(f"没有找到左侧实线，以左侧虚线作为左侧实线")
                            left_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            # 没有找到左侧实线，以左侧虚线作为左侧实线
                            # right_dashed_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            # dashed_segments.append(right_dashed_line)
                            left_solid_segments.append(left_line)
                        
                    
                    # 右侧实线：在右侧虚线右侧2.0~4.5米之间
                    if right_dashed_line is not None:
                        line_vec = right_dashed_line[1][:2] - right_dashed_line[0][:2]
                        line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-8)
                        normal_vec = np.array([-line_vec[1], line_vec[0]])
                        solid_proj = np.dot(solid_points[:, :2] - right_dashed_line[0][:2], normal_vec)
                        right_mask = (solid_proj < -2.0) & (solid_proj >= -4.5)
                        right_points = solid_points[right_mask]
                        
                        if len(right_points) >= 100:
                            right_line = fit_line_3d_ransac(right_points[:, :3])
                            right_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(right_line[0], vehicle_dir, angle_threshold=45.0)
                            right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)

                            if right_line[0] is not None:
                                right_solid_segments.append(right_line)
                                right_dashed_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)
                                # dashed_segments.append(right_dashed_line)
                                dashed_line_list.append(right_dashed_line)
                        else:
                            print(f"没有找到右侧实线，以右侧虚线作为右侧实线")
                            right_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            # 没有找到右侧实线，以右侧虚线作为右侧实线
                            # left_dashed_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            # dashed_segments.append(left_dashed_line)
                            right_solid_segments.append(right_line)
                
                if len(dashed_line_list) ==1:
                    if type(dashed_line_list[0]) == np.ndarray:
                        dashed_segments.append([dashed_line_list[0][0].tolist(), dashed_line_list[0][1].tolist()])
                    else:
                        dashed_segments.append(dashed_line_list[0])
                else:
                    dashed_segments.append(dashed_line_list)
                #  收集当前窗口的拟合结果
                current_window_segments = {
                    'dashed': dashed_line_list,
                    'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                    'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                }

            else:
                print(f"窗口 {window_id}: 虚线 (检测到 {len(lane_lines)} 条)")
                if len(lane_lines) >= 3:
                    window_center_proj = (seg_start + seg_end) / 2
                    total_proj_range = max_proj - min_proj
                    window_relative_position = (window_center_proj - min_proj) / total_proj_range
                    
                    # 判断是否在中间位置（比如在30%-70%范围内）
                    is_middle_window = 0.2 <= window_relative_position <= 0.8

                    if is_middle_window and sum(cluster_points) > len(lane_lines)*100:
                        print("处于中间位置")
                        
                    else:
                        print("不处于中间位置")
                        solid_points = window_points[window_points[:, 3] == solid_class]
                        lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=10)
                        # dashed_line = fit_line_3d_ransac(dashed_points[:, :3])
                        results_lines = determine_lane_positions(lane_lines, current_vehicle_dir)
                        left_line = results_lines[0]
                        right_line = results_lines[-1]
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                        left_solid_segments.append(left_line)
                        right_solid_segments.append(right_line)
                        if len(results_lines) == 4:
                            left_dashed_line = results_lines[1]
                            right_dashed_line = results_lines[2]
                            left_dashed_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            right_dashed_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)
                            dashed_segments.append([left_dashed_line, right_dashed_line])
                            # dashed_segments.append(right_dashed_line)
                            current_window_segments = {
                                'dashed': [left_dashed_line, right_dashed_line],
                                'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                                'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                            }
                        else:
                            dashed_line = results_lines[1]
                            dashed_line = clip_line_to_window(dashed_line, seg_start, seg_end, current_vehicle_dir)
                            if type(dashed_line) == np.ndarray:
                                dashed_segments.append([dashed_line[0].tolist(), dashed_line[1].tolist()])
                            else:
                                dashed_segments.append(dashed_line)
                            # dashed_segments.append(dashed_line)
                            current_window_segments = {
                                'dashed': [dashed_line],
                                'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                                'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                            }

                if len(lane_lines)==0:
                    print("有虚线点云，但是没有检测到虚线，开始检测到实线")
                    solid_points = window_points[window_points[:, 3] == solid_class]
                    if len(solid_points) >= 10:
                        lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=8.5)
                        results_lines = determine_lane_positions(lane_lines, current_vehicle_dir)
                        left_line = results_lines[0]
                        right_line = results_lines[-1]
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                        left_solid_segments.append(left_line)
                        right_solid_segments.append(right_line)
                        current_window_segments = {
                            'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                            'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                        }
                
                # line_result = fit_line_3d_ransac(dashed_points[:, :3])
                # if line_result[0] is not None:
                #     dashed_line = line_result[0]
                #     dashed_line = clip_line_to_window(dashed_line, seg_start, seg_end, current_vehicle_dir)
                #     dashed_segments.append(dashed_line)
                #     # 计算法向量
                #     line_vec = dashed_line[1][:2] - dashed_line[0][:2]
                #     line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-8)
                #     normal_vec = np.array([-line_vec[1], line_vec[0]])
                #     # 2. 区分左右实线
                #     solid_points = window_points[window_points[:, 3] == solid_class]
                #     if len(solid_points) >= 2:
                #         # 投影到法向量
                #         solid_proj = np.dot(solid_points[:, :2] - dashed_line[0][:2], normal_vec)
        
                #         # 左实线：在虚线左侧2.0到4.5米范围内
                #         left_mask = (solid_proj > 2.0) & (solid_proj <= 4.5)
                #         left_points = solid_points[left_mask]
                        
                #         # 右实线：在虚线右侧2.0到4.5米范围内
                #         right_mask = (solid_proj < -2.0) & (solid_proj >= -4.5)
                #         right_points = solid_points[right_mask]
                        
                #         # 左实线拟合
                #         if len(left_points) >= 100:
                #             left_line = fit_line_3d_ransac(left_points[:, :3])
                #             left_line = clip_line_to_window(left_line[0], seg_start, seg_end, current_vehicle_dir)
                #             if left_line[0] is not None:
                #                 left_solid_segments.append(left_line)
                        
                #         # 右实线拟合
                #         if len(right_points) >= 100:
                #             right_line = fit_line_3d_ransac(right_points[:, :3])
                #             right_line = clip_line_to_window(right_line[0], seg_start, seg_end, current_vehicle_dir)
                #             if right_line[0] is not None:
                #                 right_solid_segments.append(right_line)
                        
                #         # 更新当前窗口的拟合结果（超过两条虚线的情况）
                #         current_window_segments = {
                #             'dashed': [dashed_line],
                #             'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 and len(left_points) >= 100 else [],
                #             'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 and len(right_points) >= 100 else []
                #         }
                #     else:
                #         current_window_segments = {
                #             'dashed': [dashed_line],
                #             'left_solid': [],
                #             'right_solid': []
                #         }
                        

        else:
            print(f"窗口 {window_id}: 没有虚线点云")
            # 没有足够的虚线点云，需要分离不同实线车道
            solid_points = window_points[window_points[:, 3] == solid_class]
            if len(solid_points) >= 500:
                # 基于线性分布特性分离实线车道
                # lane_lines = separate_solid_lanes_by_distribution(solid_points, current_vehicle_dir)
                # print("separate_solid_lanes_by_distribution 实线数量 ", len(lane_lines))
                lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.5, max_lane_distance=8.5)
                
                if len(lane_lines) == 4:
                    print(f"窗口 {window_id}: 有四条实线")
                    # 检查当前窗口是否在中间位置
                    window_center_proj = (seg_start + seg_end) / 2
                    total_proj_range = max_proj - min_proj
                    window_relative_position = (window_center_proj - min_proj) / total_proj_range
                    
                    # 判断是否在中间位置（比如在30%-70%范围内）
                    is_middle_window = 0.2 <= window_relative_position <= 0.8
                    
                    # 检查是否有虚线点云
                    dashed_points = window_points[window_points[:, 3] == dashed_class]
                    has_dashed_points = len(dashed_points) >= 80  # 使用原有的虚线检测条件
                    
                    if not is_middle_window and not has_dashed_points:
                        # 不处于中间位置的窗口，如果检测到有三条车道线线段，那么认为中间的线是虚线
                        results_lines = determine_lane_positions(lane_lines, current_vehicle_dir)
                        left_line = results_lines[0]
                        right_line = results_lines[-1]
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                        left_solid_segments.append(left_line)
                        right_solid_segments.append(right_line)

                        left_dashed_line = results_lines[1]
                        right_dashed_line = results_lines[2]
                        left_dashed_line = clip_line_to_window(left_dashed_line, seg_start, seg_end, current_vehicle_dir)
                        right_dashed_line = clip_line_to_window(right_dashed_line, seg_start, seg_end, current_vehicle_dir)

                        dashed_segments.append([left_dashed_line, right_dashed_line])
                        
                        current_window_segments = {
                        'dashed': [left_dashed_line, right_dashed_line],
                        'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                        'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                        }
                    elif is_middle_window and not has_dashed_points:
                        lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=4.5)
                        
                        sorted_indices = np.argsort(cluster_points)
                        # lane_centers = [cluster_points[i] for i in sorted_indices]
                        lane_lines = [lane_lines[i] for i in sorted_indices]


                        results_lines = determine_lane_positions(lane_lines[-2:], current_vehicle_dir)
                        print("返回数量 ", len(results_lines))
                        left_line = results_lines[0]
                        right_line = results_lines[-1]

                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)

                        left_solid_segments.append(left_line)
    
                        right_solid_segments.append(right_line)

                        current_window_segments = {
                        'left_solid': [left_solid_segments[-1]] if len(left_solid_segments) > 0 else [],
                        'right_solid': [right_solid_segments[-1]] if len(right_solid_segments) > 0 else []
                        }
                        
                        

                if len(lane_lines) == 3:

                    # 检查当前窗口是否在中间位置
                    window_center_proj = (seg_start + seg_end) / 2
                    total_proj_range = max_proj - min_proj
                    window_relative_position = (window_center_proj - min_proj) / total_proj_range
                    
                    # 判断是否在中间位置（比如在30%-70%范围内）
                    is_middle_window = 0.2 <= window_relative_position <= 0.8
                    
                    # 检查是否有虚线点云
                    dashed_points = window_points[window_points[:, 3] == dashed_class]
                    has_dashed_points = len(dashed_points) >= 80  # 使用原有的虚线检测条件
                    
                    if is_middle_window and not has_dashed_points:
                        print("处于中间位置")
                        # 在三条线的情况下，根据两两之间的横向距离判断，筛选出两条合适的车道线
                        # 一般来说正确的两条车道线之间的距离应该在3.2米到4.5米之间
                        lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=4.5)
                        
                        sorted_indices = np.argsort(cluster_points)
                        lane_lines = [lane_lines[i] for i in sorted_indices]


                        results_lines = determine_lane_positions(lane_lines[-2:], current_vehicle_dir)
                        print("返回数量 ", len(results_lines))
                        left_line = results_lines[0]
                        right_line = results_lines[-1]
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                            
                        # 更新当前窗口的拟合结果（两条实线的情况）
                        current_window_segments = {
                        'left_solid': [left_line],
                        'right_solid': [right_line]
                        }
                    else:
                        print("不处于中间位置")
                        # 不处于中间位置的窗口，如果检测到有三条车道线线段，那么认为中间的线是虚线
                        left_line, dash_line, right_line = determine_lane_positions(lane_lines, current_vehicle_dir)
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)
                        dash_line = clip_line_to_window(dash_line, seg_start, seg_end, current_vehicle_dir)
                        if type(dash_line) == np.ndarray:
                            dashed_segments.append([dash_line[0].tolist(), dash_line[1].tolist()])
                        else:
                            dashed_segments.append(dashed_line)
                        left_solid_segments.append(left_line)
                        dashed_segments.append(dash_line)
                        right_solid_segments.append(right_line)
                    
                        # 收集当前窗口的拟合结果
                        current_window_segments = {
                            'dashed': [dash_line],
                            'left_solid': [left_solid_segments[-1]],
                            'right_solid': [right_solid_segments[-1]]
                        }

                elif len(lane_lines) == 2:
                    # 只有两条实线，需要确定左右
                    left_line, right_line = determine_lane_positions(lane_lines, current_vehicle_dir)
                    left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                    right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)

                    left_solid_segments.append(left_line)
                    right_solid_segments.append(right_line)

                    current_window_segments = {
                    'left_solid': [left_line],
                    'right_solid': [right_line]
                    }
                    
                elif len(lane_lines) == 1:
                    # 只有一条实线，需要确定是左还是右
                    # 根据位置判断
                    line = lane_lines[0]
                    line = clip_line_to_window(line, seg_start, seg_end, current_vehicle_dir)
                    line_center = (line[0][:2] + line[1][:2]) / 2
                    # 计算相对于车辆方向的横向位置
                    normal_vec = np.array([-current_vehicle_dir[1], current_vehicle_dir[0]])
                    lateral_pos = np.dot(line_center, normal_vec)
                    
                    if lateral_pos > 0:
                        # 在车辆左侧
                        left_solid_segments.append(line)
                        current_window_segments = {
                            'left_solid': [left_solid_segments[-1]],
                        }
                    else:
                        # 在车辆右侧
                        right_solid_segments.append(line)
                        current_window_segments = {
                            'right_solid': [right_solid_segments[-1]]
                        }
                
                elif len(lane_lines) == 0:
                    lane_lines, cluster_points = separate_dashed_lanes_by_fitting(solid_points, current_vehicle_dir, min_lane_distance=2.0, max_lane_distance=5.5)
                    if len(lane_lines) == 0:
                        print("没有检测到车道线")
                    else:
                        print("检测到车道线")
                        results_lines = determine_lane_positions(lane_lines, current_vehicle_dir)
                        left_line = results_lines[0]
                        right_line = results_lines[-1]
                        left_solid_segments.append(left_line)
                        right_solid_segments.append(right_line)
                        left_line = clip_line_to_window(left_line, seg_start, seg_end, current_vehicle_dir)
                        right_line = clip_line_to_window(right_line, seg_start, seg_end, current_vehicle_dir)

                        current_window_segments = {
                            'left_solid': [left_solid_segments[-1]],
                            'right_solid': [right_solid_segments[-1]]
                        }

        # print('dashed_segments {} shape {}'.format(dashed_segments, np.asarray(dashed_segments).shape))
        # print('left_solid_segments {} shape {}'.format(left_solid_segments, np.asarray(left_solid_segments).shape))
        # print('right_solid_segments {} shape {}'.format(right_solid_segments, np.asarray(right_solid_segments).shape))
        # 计算窗口中心点
        window_center_proj = (seg_start + seg_end) / 2
        window_center_2d = window_center_proj * vehicle_dir + ortho_center * np.array([-vehicle_dir[1], vehicle_dir[0]])
        window_center_3d = np.array([window_center_2d[0], window_center_2d[1], z_center])
        
        # 如果当前窗口有拟合结果，进行可视化
        # if len(window_points) >= 2 and 'current_window_segments' in locals():
        # 可视化当前窗口的拟合效果
        visualize_single_window(
            window_points, 
            current_window_segments, 
            window_center_3d, 
            current_vehicle_dir, 
            window_id,
            dashed_class, 
            solid_class, 
            window_length, 
            window_width, 
            window_height
        )
        window_id += 1
        print("==============================================================================")
    # 拼接
    # print('left_solid_segments', left_solid_segments)
    # all_detected_lanes = merge_segments_to_polyline(dashed_segments, vehicle_forward_direction[:2])
    # left_lanes = merge_segments_to_polyline(left_solid_segments, vehicle_forward_direction[:2])
    # right_lanes = merge_segments_to_polyline(right_solid_segments, vehicle_forward_direction[:2])

    # 计算平均距离
    # mean_dist = np.mean([np.dot(pt[:2] - current_center[0][:2], normal_vec) for pt in polyline])
    # results.append((polyline, mean_dist))
    # lane_lines = [lane_lines[i] for i in sorted_indices]

    # 按车辆方向对right_solid_segments和left_solid_segments进行排序
    def sort_segments_by_vehicle_direction(segments, vehicle_direction):
        """按车辆方向对线段进行排序"""
        if not segments:
            return segments
        
        # 计算每个线段的中心点在车辆方向上的投影
        segment_projections = []
        for i, segment in enumerate(segments):
            if segment is not None and len(segment) >= 2:
                # 计算线段中心点
                center_point = (segment[0][:2] + segment[1][:2]) / 2
                # 计算在车辆方向上的投影
                projection = np.dot(center_point, vehicle_direction[:2])
                segment_projections.append((projection, i, segment))
        
        # 按投影值排序（从小到大，即从车辆后方到前方）
        segment_projections.sort(key=lambda x: x[0])
        
        # 返回排序后的线段
        return [item[2] for item in segment_projections]
    
    # 对左右实线线段进行排序
    vehicle_dir_2d = vehicle_forward_direction[:2] / np.linalg.norm(vehicle_forward_direction[:2])
    right_solid_segments = sort_segments_by_vehicle_direction(right_solid_segments, vehicle_forward_direction)
    left_solid_segments = sort_segments_by_vehicle_direction(left_solid_segments, vehicle_forward_direction)

    # 汇总所有车道线（虚线+实线）
    all_lanes = []
    # 先添加右侧实线
    # for line in right_lanes:
    #     print("right line", line)
    mean_dist = 3.2
    all_lanes.append((right_solid_segments, mean_dist))
    # 添加所有虚线（按距离排序）
    # for line in all_detected_lanes:
    mean_dist = 0.0
    all_lanes.append((dashed_segments, mean_dist))
    # 添加左侧实线
    # for line in left_lanes:
    mean_dist = -3.2
    all_lanes.append((left_solid_segments, mean_dist))

    # 按法向距离排序并编号
    # all_lanes.sort(key=lambda x: x[1])
    for idx, (line, mean_dist) in enumerate(all_lanes):
        lane_type = "dashed" if mean_dist == 0.0 else "left solid" if mean_dist < 0 else "right solid"
        lane_segments.append((line, lane_type, idx))
    return lane_segments

def add_shorter_left_solid_lane(left_lanes, vehicle_forward_direction):
    if len(left_lanes)==1:
        print("左侧只有一条车道线，不进行补充")
        return []
    # 获取最左侧的两条车道线（按距离排序，距离越小越左侧）
    if np.linalg.norm(left_lanes[0][0][0] - left_lanes[0][0][-1])<np.linalg.norm(left_lanes[1][0][0] - left_lanes[1][0][-1]):
        leftmost_lane = left_lanes[0]  # 最右侧车道线
        second_left_lane = left_lanes[1]  # 第二条右侧车道线
    else:
        leftmost_lane = left_lanes[1]  # 最右侧车道线
        second_left_lane = left_lanes[0]  # 第二条右侧车道线
    left_lanes_sorted = [leftmost_lane, second_left_lane]

    print('第二条左侧车道线长度: ', np.linalg.norm(second_left_lane[0][0] - second_left_lane[0][-1]))
    print('最左侧车道线长度: ', np.linalg.norm(leftmost_lane[0][0] - leftmost_lane[0][-1]))
    
    # 获取车辆前进方向的2D向量
    vehicle_dir_2d = vehicle_forward_direction[:2] / np.linalg.norm(vehicle_forward_direction[:2])
    
    # 将最左侧车道线投影到车辆方向上
    leftmost_line = leftmost_lane[0]  # 车道线点集
    leftmost_projected = []
    for point in leftmost_line:
        # 计算点在车辆方向上的投影
        point_2d = point[:2]
        # 投影距离
        projection_distance = np.dot(point_2d, vehicle_dir_2d)
        # 投影点
        projected_point = projection_distance * vehicle_dir_2d
        leftmost_projected.append(projection_distance)
    
    # 将第二条左侧车道线投影到车辆方向上
    second_line = second_left_lane[0]  # 车道线点集
    second_projected = []
    for point in second_line:
        point_2d = point[:2]
        projection_distance = np.dot(point_2d, vehicle_dir_2d)
        second_projected.append(projection_distance)
    
    most_end_projection = max(leftmost_projected)
    if most_end_projection>(max(second_projected)+min(second_projected))/2:
        print(f"最左侧车道线位于头部")
        
        # 找到最左侧车道线的起点投影值
        leftmost_start_projection = min(leftmost_projected)
        
        # 在第二条车道线中找到比最左侧车道线起点小一些的点
        # 设置一个偏移量，比如小2米
        offset_distance = 2.0
        target_projection = leftmost_start_projection - offset_distance
        
        # 找到最接近目标投影值的点
        closest_point_idx = np.argmin(np.abs(np.array(second_projected) - target_projection))
        closest_point = second_line[closest_point_idx]
        
        # 将这个点添加到最左侧车道线的开头
        leftmost_line_updated = np.vstack([closest_point.reshape(1, -1), leftmost_line])
        
        # 更新最左侧车道线
        left_lanes_sorted[0] = (leftmost_line_updated, leftmost_lane[1])
        
        # 重新排序left_lanes
        left_lanes = left_lanes_sorted
        
        print(f"已将第二条车道线的点 {closest_point} 添加到最左侧车道线起点")
    else:
        print(f"最左侧车道线位于尾部")
        # 找到最左侧车道线的起点投影值
        leftmost_end_projection = max(leftmost_projected)
        # 在第二条车道线中找到比最左侧车道线起点小一些的点
        # 设置一个偏移量，比如小2米
        offset_distance = 2.0
        target_projection = leftmost_end_projection + offset_distance
        # 找到最接近目标投影值的点
        closest_point_idx = np.argmin(np.abs(np.array(second_projected) - target_projection))
        closest_point = second_line[closest_point_idx]
        
        # 将这个点添加到最左侧车道线的末尾
        leftmost_line_updated = np.vstack([leftmost_line, closest_point.reshape(1, -1)])
        
        # 更新最左侧车道线
        left_lanes_sorted[0] = (leftmost_line_updated, leftmost_lane[1])
        
        # 重新排序left_lanes
        left_lanes = left_lanes_sorted
        
        print(f"已将第二条车道线的点 {closest_point} 添加到最左侧车道线末尾")
    return left_lanes 

def add_shorter_right_solid_lane(right_lanes, vehicle_forward_direction):
    if len(right_lanes)==1:
        print("右侧只有一条车道线，不进行补充")
        return []
    if np.linalg.norm(right_lanes[0][0][0] - right_lanes[0][0][-1])<np.linalg.norm(right_lanes[1][0][0] - right_lanes[1][0][-1]):
        rightmost_lane = right_lanes[0]  # 最右侧车道线
        second_right_lane = right_lanes[1]  # 第二条右侧车道线
    else:
        rightmost_lane = right_lanes[1]  # 最右侧车道线
        second_right_lane = right_lanes[0]  # 第二条右侧车道线
    right_lanes_sorted = [rightmost_lane, second_right_lane]
    print('第二条右侧车道线长度: ', np.linalg.norm(second_right_lane[0][0] - second_right_lane[0][-1]))
    print('最右侧车道线长度: ', np.linalg.norm(rightmost_lane[0][0] - rightmost_lane[0][-1]))

    # 使用从右数第二条车道线的方向作为参考方向
    second_line = second_right_lane[0]  # 车道线点集
    sec_vec = second_line[-1][:2] - second_line[0][:2]
    if np.linalg.norm(sec_vec) < 1e-6:
        sec_dir_2d = vehicle_forward_direction[:2] / np.linalg.norm(vehicle_forward_direction[:2])
    else:
        sec_dir_2d = sec_vec / np.linalg.norm(sec_vec)

    # 计算相对第二条车道线方向的投影
    rightmost_line = rightmost_lane[0]
    rightmost_projected = [np.dot(point[:2], sec_dir_2d) for point in rightmost_line]
    second_projected = [np.dot(point[:2], sec_dir_2d) for point in second_line]

    # 通过端点接近性判断“头部/尾部”
    r_start2d, r_end2d = rightmost_line[0][:2], rightmost_line[-1][:2]
    s_start2d, s_end2d = second_line[0][:2], second_line[-1][:2]
    d_to_start = min(np.linalg.norm(r_start2d - s_start2d), np.linalg.norm(r_end2d - s_start2d))
    d_to_end = min(np.linalg.norm(r_start2d - s_end2d), np.linalg.norm(r_end2d - s_end2d))
    attach_to_head = d_to_start <= d_to_end

    most_end_projection = max(rightmost_projected)
    if not attach_to_head:
        print(f"最右侧车道线位于头部")
        # 找到最右侧车道线的起点投影值
        rightmost_start_projection = min(rightmost_projected)
        
        # 在第二条车道线中找到比最右侧车道线起点小一些的点
        # 设置一个偏移量，比如小2米
        offset_distance = 2.0
        target_projection = rightmost_start_projection - offset_distance
        
        # 找到最接近目标投影值的点
        closest_point_idx = np.argmin(np.abs(np.array(second_projected) - target_projection))
        closest_point = second_line[closest_point_idx]
        
        # 将这个点添加到最右侧车道线的开头
        rightmost_line_updated = np.vstack([closest_point.reshape(1, -1), rightmost_line])
        
        # 更新最右侧车道线
        right_lanes_sorted[0] = (rightmost_line_updated, rightmost_lane[1])
        
        # 重新排序right_lanes
        right_lanes = right_lanes_sorted
        
        print(f"已将第二条车道线的点 {closest_point} 添加到最右侧车道线起点")
        
    else:
        print(f"最右侧车道线位于尾部")
        
        # 找到最右侧车道线的终点投影值
        rightmost_end_projection = max(rightmost_projected)
        
        # 在第二条车道线中找到比最右侧车道线终点大一些的点（沿第二条线方向）
        offset_distance = 2.0
        target_projection = rightmost_end_projection + offset_distance
        
        # 找到最接近目标投影值的点
        closest_point_idx = np.argmin(np.abs(np.array(second_projected) - target_projection))
        closest_point = second_line[closest_point_idx]
        
        # 将这个点添加到最左侧车道线的开头
        rightmost_line_updated = np.vstack([rightmost_line, closest_point.reshape(1, -1)])
        
        # 更新最右侧车道线
        right_lanes_sorted[0] = (rightmost_line_updated, rightmost_lane[1])
        
        # 重新排序right_lanes
        right_lanes = right_lanes_sorted
        
        print(f"已将第二条车道线的点 {closest_point} 添加到最右侧车道线终点")
    return right_lanes

def process_lane_segment(segment_dir, segment_id, min_displacement=50.0, visualize=False, visualize_3d=False, arrow_processor=None):
    """
    处理单个车道线段，生成矢量化表示并显示三维可视化
    
    参数:
        segment_dir: 段数据目录
        segment_id: 段ID
        min_displacement: 最小位移阈值（米）
        visualize: 是否可视化
        visualize_3d: 是否使用3D可视化
        arrow_processor: 可选箭头回调，不改变车道线返回格式
        
    返回:
        lane_segments: 车道线段列表，每个元素是(线段点, 类型, 编号)
        stop_line_pos: 停止线位置
        lane_count: 车道数量
    """
    # 读取点云和位姿
    pcd_file = os.path.join(segment_dir, f"lane_segment_{segment_id:03d}.pcd")
    pose_file = os.path.join(segment_dir, f"lane_segment_{segment_id:03d}_poses.txt")
    
    points = load_point_cloud(pcd_file)
    poses = load_poses(pose_file)
    
    if points is None or len(poses) == 0:
        print(f"段 {segment_id}: 点云或位姿数据为空")
        return [], None, 0

    # 计算位移
    total_displacement = calculate_pose_displacement(poses)
    if total_displacement < min_displacement:
        print(f"段 {segment_id}: 位移 {total_displacement:.2f}m 小于阈值 {min_displacement}m，跳过")
        return [], None, 0
    
    # 提取车道线点云
    solid_lane_points = filter_points_by_class(points, [LANE_SOLID_CLASS])
    dashed_lane_points = filter_points_by_class(points, [LANE_DASHED_CLASS])
    stop_line_points = filter_points_by_class(points, [STOP_LINE_CLASS])
    
    print(f"段 {segment_id}: 实线点数={len(solid_lane_points)}, 虚线点数={len(dashed_lane_points)}, 停止线点数={len(stop_line_points)}")

    # 提取停止线位置
    stop_line_pos = None
    if len(stop_line_points) >= 2:
        stop_line_pos = np.mean(stop_line_points[:, :3], axis=0)
    

    correct_direction = None
    # 使用第一个位姿作为参考
    reference_pose = poses[0][1]  # 第一个位姿的数据部分
    
    # 计算车辆前进方向（优先使用位姿四元数，更准确）
    x, y, z, qx, qy, qz, qw = reference_pose
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    vehicle_forward_direction = rotation.apply([1, 0, 0])  # 车辆前进方向（X轴正方向）
    vehicle_forward_direction = vehicle_forward_direction / np.linalg.norm(vehicle_forward_direction)
    print(f"段 {segment_id}: 车辆前进方向向量 = {vehicle_forward_direction}")
    
    # 同时计算轨迹方向作为参考
    if len(poses) >= 2:
        # 假设poses为N×(timestamp, [x, y, z, ...])
        trajectory_points = np.array([p[1][:3] for p in poses])

        # 计算PCA主方向
        centroid = np.mean(trajectory_points, axis=0)
        centered_points = trajectory_points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 修正PCA方向：确保方向与实际轨迹方向一致
        # 计算轨迹的起点到终点的方向作为参考
        start_pos = trajectory_points[0]
        end_pos = trajectory_points[-1]
        reference_direction = end_pos - start_pos
        reference_direction = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)
        
        # 计算PCA方向与参考方向的内积
        dot_product = np.dot(principal_direction, reference_direction)
        
        # 如果内积为负，说明PCA方向与参考方向相反，需要翻转
        if dot_product < 0:
            principal_direction = -principal_direction
            print(f"段 {segment_id}: PCA方向已翻转，确保与实际轨迹方向一致")
        
        trajectory_direction = principal_direction / (np.linalg.norm(principal_direction) + 1e-8)

        # start_pos = np.array(poses[0][1][:3])
        # end_pos = np.array(poses[-1][1][:3])
        # trajectory_direction = end_pos - start_pos
        # trajectory_direction = trajectory_direction / np.linalg.norm(trajectory_direction)
        print(f"段 {segment_id}: 轨迹方向向量 = {trajectory_direction}")
        
        # 计算两个方向的夹角
        angle_diff = np.arccos(np.clip(np.dot(vehicle_forward_direction, trajectory_direction), -1.0, 1.0))
        angle_diff_deg = np.degrees(angle_diff)
        print(f"段 {segment_id}: 车辆前进方向与轨迹方向夹角 = {angle_diff_deg:.2f}度")
        correct_direction = trajectory_direction
        # 如果夹角太大，使用轨迹方向
        if angle_diff_deg > 10:
            print(f"段 {segment_id}: 夹角过大，使用轨迹方向作为前进方向")
            # vehicle_forward_direction = trajectory_direction
            correct_direction = trajectory_direction
    else:
        # trajectory_direction = vehicle_forward_direction
        correct_direction = vehicle_forward_direction
    
    # 检查是否有足够的虚线点
    lane_segments = []
    lane_count = 0
    
    if (len(dashed_lane_points) > 200):
        print(f"段 {segment_id}: 虚线点数充足，开始拟合")
        
        # 智能扩展虚线点云（包含两端实线段）
        classified_points, nonlinearity_score, direction_change_std = classify_dashed_lane_with_ends(
            points, poses, trajectory_direction, 
            dashed_class=LANE_DASHED_CLASS, 
            solid_class=LANE_SOLID_CLASS
        )
        if (nonlinearity_score>17 and nonlinearity_score<30) or direction_change_std>1.2:
            print("非线性离散度过高，采用分段线性化拟合！")
            lane_segments = sliding_window_lane_fitting(points, correct_direction, dashed_class=LANE_DASHED_CLASS, solid_class=LANE_SOLID_CLASS, window_length=20.0, window_width=20.0, window_height=10.0)
            # if visualize and visualize_3d:
                # visualize_segment_3d_with_points(points, lane_segments, stop_line_pos, segment_id, correct_direction)

            if arrow_processor is not None:
                arrow_processor(lane_segments, filter_points_by_class(points, [LANE_ARROW_CLASS]),
                                segment_id, correct_direction, fitting_type=0)
            return lane_segments, 0, 3
        elif nonlinearity_score>50:
            print("很可能存在转弯，请先使用road_structure_analyer.py分段再进行处理！")
            return [], None, 0
        # 用扩展后的点云继续后续处理
        dashed_lane_points = filter_points_by_class(classified_points, [LANE_DASHED_CLASS])
        
        # 拟合虚线（直接使用RANSAC拟合，不进行投影）
        dashed_line, _ = fit_line_ransac(dashed_lane_points)
        if dashed_line is not None:
            # lane_segments.append((dashed_line, "dashed", 1))
            print(f"段 {segment_id}: 成功拟合虚线")
            
            # 根据虚线拟合结果调整车辆前进方向
            # 计算虚线的实际方向
            if len(dashed_line) >= 2:
                dashed_start = dashed_line[0][:2]  # 取前两个坐标
                dashed_end = dashed_line[-1][:2]
                dashed_direction = dashed_end - dashed_start
                dashed_direction = dashed_direction / np.linalg.norm(dashed_direction)
                
                # 检查虚线方向与车辆前进方向的一致性
                dot_product = np.dot(dashed_direction, correct_direction[:2])
                if dot_product < 0:
                    # 如果方向相反，翻转虚线方向
                    dashed_direction = -dashed_direction
                
                # 使用虚线方向作为调整后的车辆前进方向
                adjusted_forward_direction = np.array([dashed_direction[0], dashed_direction[1], 0])
                print(f"段 {segment_id}: 调整前进方向从 {correct_direction[:2]} 到 {adjusted_forward_direction[:2]}")
                
                # 更新车辆前进方向
                vehicle_forward_direction = adjusted_forward_direction
        else:
            print(f"段 {segment_id}: 虚线拟合失败")

        # 使用拟合后的虚线直线段来分离多条实线（支持多车道）
        if dashed_line is not None and len(dashed_line) >= 2:
            dashed_start = dashed_line[0][:2]
            dashed_end = dashed_line[-1][:2]
            dashed_direction = dashed_end - dashed_start
            dashed_direction = dashed_direction / np.linalg.norm(dashed_direction)
            dashed_normal = np.array([-dashed_direction[1], dashed_direction[0]])  # 指向左侧的法向量

            # 分块拟合
            dashed_segments = fit_lane_segments_adaptive(dashed_lane_points, dashed_normal, block_size=20.0, min_points=150)
            # 可视化分块线段和点云
            # dash_global_line = robust_fit_line(dashed_lane_points, dashed_normal, eps=0.5, min_samples=500, ransac_thresh=0.2)
            dash_global_line, _ = fit_line_ransac(dashed_lane_points)
            dash_global_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(dash_global_line, dashed_normal, angle_threshold=45.0)

            visualize_segment_3d_with_points(dashed_lane_points, [], None, segment_id=f"block_{1}", vehicle_forward_direction=vehicle_forward_direction, block_segments=dashed_segments, global_line=dash_global_line)

            # 合并分块线段为一条车道线
            improved_dashed_line = merge_segments_to_polyline(dashed_segments, dashed_normal)
            dashed_length = np.linalg.norm(improved_dashed_line[0][:2] - improved_dashed_line[-1][:2])
            print(f"段 {segment_id}: 找到第一条虚线车道线")
            # print("improved_dashed_line shape: ", np.array(improved_dashed_line).shape)
            # visualize_lane_segments_3d(dashed_segments, [], title="所有段的车道线和停止线")
            # 将第一条虚线加入结果

            # 使用第一条虚线分离实线
            min_points = 800
            # 递归分离左侧所有虚线
            print('开始向左分离虚线')
            left_dashed_lanes = fit_multi_lanes(
                dashed_lane_points, improved_dashed_line, normal_vec=dashed_normal,
                solid_dashed_points=solid_lane_points, dashed_length=dashed_length, class_type=LANE_DASHED_CLASS,
                min_dist=2.8, max_dist=4.5, min_points=min_points, direction=-1
            )
            # 递归分离右侧所有虚线
            print('开始向右分离虚线')
            right_dashed_lanes = fit_multi_lanes(
                dashed_lane_points, improved_dashed_line, dashed_normal,
                solid_dashed_points=solid_lane_points, dashed_length=dashed_length, class_type=LANE_DASHED_CLASS,
                min_dist=2.8, max_dist=4.5, min_points=min_points, direction=1
            )
            
            print(f"段 {segment_id}: 分离出 {len(left_dashed_lanes)} 条左侧虚线，{len(right_dashed_lanes)} 条右侧虚线")
            all_detected_lanes = []
            # # 递归分离左右两侧的其他虚线车道线
            # def recursive_fit_dashed_lanes(center_line, normal_vec, remaining_points, direction, lane_id):
            #     """递归分离虚线车道线（增强：平行性+横向距离约束）"""
            #     if len(remaining_points) < 200:  # 点数不足
            #         return []
                
            #     # 使用当前虚线分离其他虚线
            #     dashed_lanes = fit_multi_lanes(
            #         remaining_points, center_line, normal_vec,
            #         min_dist=2.5, max_dist=4.5, min_points=200, direction=direction
            #     )
                
            #     detected_lanes = []
            #     for line, mean_dist in dashed_lanes:
            #         if len(line) >= 2:
            #             # 对分离出的虚线进行分块拟合
            #             line_segments = fit_lane_segments_adaptive(line, vehicle_forward_direction[:2], block_size=20.0, min_points=150)
            #             if len(line_segments) > 0:
            #                 # 合并为polyline
            #                 polyline = merge_segments_to_polyline(line_segments, vehicle_forward_direction[:2])
            #                 if polyline.shape[0] >= 2:
            #                     # --- 新增平行性和横向距离判断 ---
            #                     # 1. 拟合当前polyline的主方向
            #                     cur_dir = polyline[-1][:2] - polyline[0][:2]
            #                     cur_dir = cur_dir / (np.linalg.norm(cur_dir) + 1e-8)
            #                     # 2. 拟合center_line的主方向
            #                     ref_dir = center_line[-1][:2] - center_line[0][:2]
            #                     ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-8)
            #                     # 3. 方向一致性
            #                     cos_angle = np.abs(np.dot(cur_dir, ref_dir))
            #                     # 4. 横向距离
            #                     lateral_dist = line_to_line_lateral_distance(polyline[:2], center_line[:2])
            #                     print(f"递推分离虚线: cos_angle={cos_angle:.3f}, lateral_dist={lateral_dist:.2f}")
            #                     if cos_angle > 0.98 and lateral_dist > 2.0:
            #                         detected_lanes.append((polyline, mean_dist, f"dashed_{lane_id}"))
            #                         print(f"段 {segment_id}: 分离出第 {lane_id} 条虚线车道线，点数={len(line)}，横向距离={lateral_dist:.2f}")
            #                     else:
            #                         print(f"段 {segment_id}: 拒绝分离出的虚线，方向一致性={cos_angle:.3f}，横向距离={lateral_dist:.2f}")
            #     return detected_lanes
            
            # # 递归分离左侧虚线
            # left_dashed_lanes = recursive_fit_dashed_lanes(
            #     improved_dashed_line, dashed_normal, remaining_dashed_points, 1, 1
            # )
            
            # # 递归分离右侧虚线
            # right_dashed_lanes = recursive_fit_dashed_lanes(
            #     improved_dashed_line, dashed_normal, remaining_dashed_points, -1, 2
            # )
            
            # # 汇总所有虚线车道线
            # all_detected_lanes.extend(left_dashed_lanes)
            # all_detected_lanes.extend(right_dashed_lanes)
            
            # print(f"段 {segment_id}: 总共分离出 {len(all_detected_lanes)} 条虚线车道线")
            
            min_points = 720
            if len(left_dashed_lanes) > 0:
                # 递归分离左侧所有实线
                left_lanes = fit_multi_lanes(
                    solid_lane_points, left_dashed_lanes[0][0], dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=-1
                )
                all_detected_lanes.append((improved_dashed_line, 0.0, "dashed_0"))
                all_detected_lanes.append((left_dashed_lanes[0][0], 0.0, "dashed_1"))

                # 递归分离右侧所有实线
                right_lanes = fit_multi_lanes(
                    solid_lane_points, improved_dashed_line, dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=1
                )
                left_length = np.linalg.norm(left_lanes[0][0][0] - left_lanes[0][0][-1])
                print(f"left_dashed_lanes: 左侧实线长度: {left_length:.2f}米")
                right_length = np.linalg.norm(right_lanes[0][0][0] - right_lanes[0][0][-1])
                print(f"left_dashed_lanes: 右侧实线长度: {right_length:.2f}米")
                if left_length<100:
                    if len(left_lanes) == 1:
                        left_lanes.append((left_dashed_lanes[0][0], 0.0))
                        print("left_dashed_lanes: 左侧仅一条实线，加入当前虚线形成两条车道线")
                    left_lanes = add_shorter_left_solid_lane(left_lanes, vehicle_forward_direction)
                if right_length<100:
                    if len(right_lanes)==1:
                        right_lanes.append((improved_dashed_line, 0.0))
                    right_lanes = add_shorter_right_solid_lane(right_lanes, vehicle_forward_direction)
            
            elif len(right_dashed_lanes) > 0:
                # 递归分离右侧所有实线
                right_lanes = fit_multi_lanes(
                    solid_lane_points, right_dashed_lanes[0][0], dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=1
                )
                all_detected_lanes.append((improved_dashed_line, 0.0, "dashed_0"))
                all_detected_lanes.append((right_dashed_lanes[0][0], 0.0, "dashed_1"))
                left_lanes = fit_multi_lanes(
                    solid_lane_points, improved_dashed_line, dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=-1
                )
                left_length = np.linalg.norm(left_lanes[0][0][0] - left_lanes[0][0][-1])
                print(f"right_dashed_lanes: 左侧实线长度: {left_length:.2f}米")
                right_length = np.linalg.norm(right_lanes[0][0][0] - right_lanes[0][0][-1])
                print(f"right_dashed_lanes: 右侧实线长度: {right_length:.2f}米")
                if left_length<100:
                    if len(left_lanes) == 1:
                        left_lanes.append((improved_dashed_line, 0.0))
                        print("right_dashed_lanes: 左侧仅一条实线，加入当前虚线形成两条车道线")
                    left_lanes = add_shorter_left_solid_lane(left_lanes, vehicle_forward_direction)
                if right_length<100:
                    if len(right_lanes) == 1:
                        right_lanes.append((right_dashed_lanes[0][0], 0.0))
                        print("right_dashed_lanes: 右侧仅一条实线，加入当前虚线形成两条车道线")
                    right_lanes = add_shorter_right_solid_lane(right_lanes, vehicle_forward_direction)
            else:
                left_lanes = fit_multi_lanes(
                    solid_lane_points, improved_dashed_line, dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=-1
                )
                right_lanes = fit_multi_lanes(
                    solid_lane_points, improved_dashed_line, dashed_normal,
                    min_dist=2.8, max_dist=4.5, min_points=min_points, direction=1
                )
                print('左侧实线车道线条数: ', len(left_lanes))
                print('右侧实线车道线条数: ', len(right_lanes))

                if len(left_lanes) >1 and np.linalg.norm(left_lanes[1][0][0] - left_lanes[1][0][-1])<np.linalg.norm(left_lanes[0][0][0] - left_lanes[0][0][-1])/2:
                    # 将最左侧的两条车道线投影在车辆方向上，然后找到比最左侧车道线的起点小一些的从左边数第二条车道线的的点，将这个点添加到最左侧车道线，作为起点
                    print(f"段 {segment_id}: 左侧有多条车道线，进行车道线起点调整")
                    left_lanes = add_shorter_left_solid_lane(left_lanes, vehicle_forward_direction)                  
                
                    
                if len(right_lanes) >1 and np.linalg.norm(right_lanes[1][0][0] - right_lanes[1][0][-1])<np.linalg.norm(right_lanes[0][0][0] - right_lanes[0][0][-1])/2:
                    print(f"段 {segment_id}: 右侧有多条车道线，进行车道线起点调整")
                    right_lanes = add_shorter_right_solid_lane(right_lanes, vehicle_forward_direction)

                all_detected_lanes.append((improved_dashed_line, 0.0, "dashed_0"))
            
            # 汇总所有车道线（虚线+实线）
            all_lanes = []
            # 先添加右侧实线
            for line, mean_dist in right_lanes:
                print("right_lanes shape: ", np.array(line).shape)
                all_lanes.append((line, 3.2))
            # 添加所有虚线（按距离排序）
            for line, mean_dist, lane_type in all_detected_lanes:
                print("all_detected_lanes shape: ", np.array(line).shape)
                all_lanes.append((line, 0.0))
            # 添加左侧实线
            for line, mean_dist in left_lanes:
                print("left_lanes shape: ", np.array(line).shape)
                all_lanes.append((line, -3.2))

            # 按法向距离排序并编号
            all_lanes.sort(key=lambda x: x[1])
            for idx, (line, mean_dist) in enumerate(all_lanes):
                fallback_type = "dashed" if mean_dist == 0.0 else "solid"
                lane_type = classify_lane_type_by_semantic_ratio(
                    line,
                    solid_lane_points,
                    dashed_lane_points,
                    solid_threshold=0.8,
                    search_radius=0.6,
                    fallback_type=fallback_type
                )
                lane_segments.append((line, lane_type, idx))
            print(f"段 {segment_id}: 共检测到{len(all_lanes)}条车道线（含虚线）")
        else:
            print(f"段 {segment_id}: 虚线点数不足，无法拟合")
        # 将检测到的所有车道线加入lane_segments
    # for idx, (line, mean_dist, lane_type) in enumerate(all_detected_lanes):
    #     lane_type = "dashed" if mean_dist == 0.0 else "solid"
    #     lane_segments.append((line, lane_type, idx))

    # print(f"段 {segment_id}: 总共检测到 {len(all_detected_lanes)} 条车道线")
    
    # 可视化最终结果
    if len(lane_segments) > 0:
        visualize_segment_3d_with_points(points, lane_segments, stop_line_pos, segment_id, vehicle_forward_direction)
    
    # 统一车道线长度
    if len(lane_segments) > 1:
        print(f"段 {segment_id}: 统一车道线长度")
        lane_segments = normalize_lane_line_length(lane_segments, vehicle_forward_direction, dashed_length)
    
    lane_count = 3 if (len(lane_segments) == 3) else len(lane_segments)

    if arrow_processor is not None and lane_segments:
        arrow_processor(lane_segments, filter_points_by_class(points, [LANE_ARROW_CLASS]),
                        segment_id, correct_direction, solid_lane_points, improved_dashed_line)
    
    # 以实线为例，进行分块和整体拟合可视化
    # block_segments = None
    # global_line = None
    # if len(solid_lane_points) >= 2:
    #     # 整体拟合
    #     global_line = robust_fit_line(solid_lane_points, vehicle_forward_direction[:2], eps=0.5, min_samples=5, ransac_thresh=0.2)
    #     # 分块拟合
    #     block_segments = fit_lane_segments_adaptive(solid_lane_points, vehicle_forward_direction[:2], block_size=20.0, min_points=150)

    # 三维可视化
    if visualize and visualize_3d:
        visualize_segment_3d_with_points(points, lane_segments, stop_line_pos, segment_id, vehicle_forward_direction)
    
    print(f"段 {segment_id}: 处理完成，输出 {len(lane_segments)} 条车道线")
    return lane_segments, 1, lane_count

def visualize_segment_3d_with_points(original_points, lane_segments, stop_line_pos, segment_id, vehicle_forward_direction=None, block_segments=None, global_line=None):
    """
    三维可视化单个段的车道线拟合结果和原始点云
    参数:
        original_points: 原始点云数据
        lane_segments: 拟合的车道线段列表
        stop_line_pos: 停止线位置
        segment_id: 段ID
        block_segments: 分块线段列表（可选，List of (2,3) ndarray）
        global_line: 整体拟合线段（可选，(2,3) ndarray）
    """
    # 创建可视化窗口
    window_title = f"段 {segment_id}: 车道线拟合结果和原始点云"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1400, height=900)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # 添加原始点云（按类别着色）
    if original_points is not None and len(original_points) > 0:
        # 创建点云对象
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(original_points[:, :3])
        
        # 根据类别设置颜色
        colors = np.zeros((len(original_points), 3))
        for i, point in enumerate(original_points):
            if len(point) > 3:  # 确保有类别信息
                if point[3] == LANE_SOLID_CLASS:  # 实线
                    colors[i] = [0, 0.5, 0]  # 暗绿色
                elif point[3] == LANE_DASHED_CLASS:  # 虚线
                    colors[i] = [0.5, 0, 0]  # 暗红色
                elif point[3] == STOP_LINE_CLASS:  # 停止线
                    colors[i] = [0, 0, 0.5]  # 暗蓝色
                else:
                    colors[i] = [0.3, 0.3, 0.3]  # 灰色
            else:
                colors[i] = [0.3, 0.3, 0.3]  # 灰色
        
        raw_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(raw_pcd)
    
    def draw_arraw(segment):
        # 添加端点球体
        for point in segment:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            vis.add_geometry(sphere)
        
        # 添加虚线方向箭头
        # 计算虚线方向
        # 计算虚线方向向量
        line_vec = segment[1] - segment[0]
        line_length = np.linalg.norm(line_vec)
        if line_length > 1e-8:
            line_direction = line_vec / line_length
            
            # 计算箭头位置（在线段中点）
            arrow_position = (segment[0] + segment[1]) / 2
            
            # 箭头参数
            arrow_length = 2.0  # 箭头长度
            arrow_head_length = 0.8  # 箭头头部长度
            arrow_head_width = 0.6   # 箭头头部宽度
            
            # 计算箭头终点
            arrow_end = arrow_position + line_direction * arrow_length
            
            # 创建箭头线段
            arrow_line = o3d.geometry.LineSet()
            arrow_line.points = o3d.utility.Vector3dVector([arrow_position, arrow_end])
            arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            arrow_line.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])  # 橙色
            vis.add_geometry(arrow_line)
            
            # 计算垂直于箭头方向的向量
            perp_dir = np.array([-line_direction[1], line_direction[0], 0])
            
            # 箭头头部的三个顶点
            arrow_tip = arrow_end
            arrow_base = arrow_end - line_direction * arrow_head_length
            
            # 箭头头部的左右两个顶点
            arrow_left = arrow_base + perp_dir * arrow_head_width
            arrow_right = arrow_base - perp_dir * arrow_head_width
            
            # 创建箭头头部的三角形
            arrow_head = o3d.geometry.TriangleMesh()
            arrow_head.vertices = o3d.utility.Vector3dVector([arrow_tip, arrow_left, arrow_right])
            arrow_head.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
            arrow_head.paint_uniform_color([1, 0.5, 0])  # 橙色
            vis.add_geometry(arrow_head)

    # 添加拟合的车道线
    dashed_direction = None  # 用于存储虚线方向
    # print("lane_segments shape", len(lane_segments))
    for i, segment_info in enumerate(lane_segments):
        # 检查segment_info的格式
        if isinstance(segment_info, tuple):
            if len(segment_info) >= 2:
                segment = segment_info[0]
                lane_type = segment_info[1]
                # print("segment_info", segment_info)
            else:
                segment = segment_info
                lane_type = "unknown"
        else:
            segment = segment_info
            lane_type = "unknown"

        if type(segment) == list:
            segment = np.array(segment)
        
        # print("segment: ", segment)
        # 确保点云有三维坐标
        if segment.shape[1] == 2:
            # 如果只有XY坐标，添加Z坐标（高度）
            z_coords = np.zeros((segment.shape[0], 1))
            segment_3d = np.hstack((segment, z_coords))
        else:
            segment_3d = segment[:, :3]
        
        # 创建线段
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(segment_3d)
        
        # 创建连接线
        lines = []
        for j in range(len(segment_3d) - 1):
            lines.append([j, j + 1])
        # print("lines: ", lines)
        
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # 根据车道线类型设置颜色
        if lane_type == "solid":
            # 实线 - 亮绿色 [0, 1, 0]
            colors = np.array([[0, 1, 0] for _ in range(len(lines))])
        elif lane_type == "dashed":
            # 虚线 - 亮红色 [1, 0, 0]
            colors = np.array([[1, 0, 0] for _ in range(len(lines))])
            # 计算虚线方向（使用调整后的车辆前进方向）
            if len(segment_3d) >= 2 and vehicle_forward_direction is not None:
                # 使用虚线的中点作为起点
                mid_idx = len(segment_3d) // 2
                start_point = segment_3d[mid_idx]
                # 使用调整后的车辆前进方向
                forward_direction_3d = np.array([vehicle_forward_direction[0], vehicle_forward_direction[1], 0])
                forward_direction_3d = forward_direction_3d / np.linalg.norm(forward_direction_3d)
                dashed_direction = (start_point, forward_direction_3d)
        else:
            # 未知类型 - 亮灰色 [0.7, 0.7, 0.7]
            colors = np.array([[0.7, 0.7, 0.7] for _ in range(len(lines))])
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
        
        # 添加拟合线的点云可视化（更大的点）
        fit_pcd = o3d.geometry.PointCloud()
        fit_pcd.points = o3d.utility.Vector3dVector(segment_3d)
        if lane_type == "solid":
            fit_pcd.paint_uniform_color([0, 1, 0])  # 亮绿色
        elif lane_type == "dashed":
            fit_pcd.paint_uniform_color([1, 0, 0])  # 亮红色
        else:
            fit_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 亮灰色
        vis.add_geometry(fit_pcd)
    
    # 可视化车辆前进方向（虚线方向）
    if vehicle_forward_direction is not None:
        # 使用虚线的中点作为箭头起点
        if len(lane_segments) > 0:
            # 找到虚线
            dashed_segment = None
            for segment_info in lane_segments:
                if isinstance(segment_info, tuple) and len(segment_info) >= 2:
                    if segment_info[1] == "dashed":
                        dashed_segment = segment_info[0]
                        break
            
            if dashed_segment is not None and len(dashed_segment) > 0:
                # 使用虚线的中点
                mid_idx = len(dashed_segment) // 2
                start_point = dashed_segment[mid_idx][:3]
            else:
                # 如果没有虚线，使用原点
                start_point = np.array([0, 0, 0])
        else:
            start_point = np.array([0, 0, 0])
        
        # 创建明显的车辆前进方向箭头
        arrow_length = 5.0  # 箭头长度
        forward_direction_3d = np.array([vehicle_forward_direction[0], vehicle_forward_direction[1], 0])
        forward_direction_3d = forward_direction_3d / np.linalg.norm(forward_direction_3d)
        arrow_end = start_point + forward_direction_3d * arrow_length
        
        # 创建箭头主体（粗线段）
        arrow_points = o3d.utility.Vector3dVector([start_point, arrow_end])
        arrow_lines = o3d.utility.Vector2iVector([[0, 1]])
        
        arrow_line_set = o3d.geometry.LineSet()
        arrow_line_set.points = arrow_points
        arrow_line_set.lines = arrow_lines
        arrow_line_set.paint_uniform_color([1, 1, 0])  # 亮黄色
        
        vis.add_geometry(arrow_line_set)
        
        # 创建箭头头部（大圆锥）
        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.8, height=1.5)
        
        # 计算圆锥的旋转角度，使其指向前进方向
        # 计算圆锥应该指向的方向向量
        target_direction = forward_direction_3d
        
        # 圆锥默认指向Z轴正方向，需要旋转到目标方向
        # 计算旋转轴和角度
        default_direction = np.array([0, 0, 1])  # 圆锥默认方向
        rotation_axis = np.cross(default_direction, target_direction)
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(default_direction, target_direction)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 创建旋转矩阵
            rotation_matrix = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            cone.rotate(rotation_matrix, center=[0, 0, 0])
        
        # 将圆锥移动到箭头末端
        cone.translate(arrow_end)
        cone.paint_uniform_color([1, 1, 0])  # 亮黄色
        
        vis.add_geometry(cone)
        
        # 添加文字标签
        label_point = start_point + forward_direction_3d * (arrow_length + 2.0)
        # 创建一个小球体作为标签位置标记
        label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        label_sphere.translate(label_point)
        label_sphere.paint_uniform_color([1, 1, 0])  # 亮黄色
        vis.add_geometry(label_sphere)
        
        print(f"段 {segment_id}: 车辆前进方向已可视化 (亮黄色箭头)")
        print(f"段 {segment_id}: 前进方向向量 = {vehicle_forward_direction}")
    
    # 添加停止线
    if stop_line_pos is not None:
        # 创建停止线标记（球体）
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        # 确保stop_line_pos是numpy数组且形状正确
        if isinstance(stop_line_pos, (list, tuple)):
            stop_line_pos = np.array(stop_line_pos)
        
        # 检查stop_line_pos是否为空或形状不正确
        if stop_line_pos.size == 0:
            print(f"段 {segment_id}: 停止线位置为空，跳过可视化")
        else:
            if stop_line_pos.ndim == 1:
                if len(stop_line_pos) >= 3:
                    stop_line_pos = stop_line_pos[:3].reshape(3, 1)
                else:
                    print(f"段 {segment_id}: 停止线位置维度不足，跳过可视化")
                    return
            
            sphere.translate(stop_line_pos)
            sphere.paint_uniform_color([0, 0, 1])  # 蓝色
            vis.add_geometry(sphere)
    
    # ====== 新增：分块线段可视化 ======
    if block_segments is not None and len(block_segments) > 0:
        for seg in block_segments:
            # print("分块线段可视化 ", seg)
            draw_arraw(seg)
            seg = np.asarray(seg)
            if seg.shape[1] == 2:
                z_coords = np.zeros((seg.shape[0], 1))
                seg_3d = np.hstack((seg, z_coords))
            else:
                seg_3d = seg[:, :3]
            line_set = o3d.geometry.LineSet()
            points = o3d.utility.Vector3dVector(seg_3d)
            lines = [[0, 1]]
            line_set.points = points
            line_set.lines = o3d.utility.Vector2iVector(lines)
            # 蓝色
            colors = np.array([[0, 0.5, 1]])
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)
    # ====== 新增：整体拟合线段可视化 ======
    if global_line is not None:
        if global_line.shape[1] == 2:
            z_coords = np.zeros((global_line.shape[0], 1))
            global_3d = np.hstack((global_line, z_coords))
        else:
            global_3d = global_line[:, :3]
        draw_arraw(global_line)
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(global_3d)
        lines = [[0, 1]]
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # 黄色
        colors = np.array([[1, 1, 0]])
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
    # ====== 新增：最终融合车道线可视化 ======
    # lane_segments中每个元素是(polyline, mean_dist)，polyline为(N,3)
    # if lane_segments is not None and len(lane_segments) > 0:
    #     for i, segment_info in enumerate(lane_segments):
    #         if isinstance(segment_info, tuple):
    #             polyline = segment_info[0]
    #         else:
    #             polyline = segment_info
    #         if polyline is not None and len(polyline) > 1:
    #             polyline = np.asarray(polyline)
    #             if polyline.shape[1] == 2:
    #                 z_coords = np.zeros((polyline.shape[0], 1))
    #                 polyline_3d = np.hstack((polyline, z_coords))
    #             else:
    #                 polyline_3d = polyline[:, :3]
    #             line_set = o3d.geometry.LineSet()
    #             points = o3d.utility.Vector3dVector(polyline_3d)
    #             lines = [[j, j + 1] for j in range(len(polyline_3d) - 1)]
    #             line_set.points = points
    #             line_set.lines = o3d.utility.Vector2iVector(lines)
    #             # 紫色显示最终融合车道线
    #             colors = np.array([[0.7, 0.2, 1.0] for _ in range(len(lines))])
    #             line_set.colors = o3d.utility.Vector3dVector(colors)
    #             vis.add_geometry(line_set)

    # 设置视角
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 3.0  # 原始点云点大小
    opt.line_width = 5.0  # 线宽
    
    # 设置相机视角
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # 运行可视化
    # print(f"显示段 {segment_id} 的三维可视化窗口..., block_segments个数 {len(block_segments)}")
    vis.run()
    vis.destroy_window()

def enu_to_wgs84(lat_ref, lon_ref, h_ref, segment):
    """
    将 ENU 坐标转换为 WGS84 坐标。
    参数：
        lat_ref, lon_ref, h_ref: 参考点的纬度（度）、经度（度）、高度（米）
        east, north, up: ENU 坐标（东、北、上，单位：米）
    返回：
        (lat, lon, h): 目标点的 WGS84 坐标（纬度、经度、高度）
    """
    # 使用 WGS84 椭球模型
    from geographiclib.geodesic import Geodesic
    geod = Geodesic.WGS84
    
    east = segment[0]
    north = segment[1]
    up = segment[2]
    # 计算地表距离和方位角
    distance = math.sqrt(east**2 + north**2)  # 地表距离（米）
    if distance == 0:
        azimuth = 0  # 如果距离为 0，方位角无意义，设为 0
    else:
        azimuth = math.degrees(math.atan2(east, north))  # 方位角（度）
    
    # 确保方位角在 [0, 360) 范围内
    azimuth = azimuth % 360
    
    # 使用 Direct 方法计算目标点的经纬度
    result = geod.Direct(lat_ref, lon_ref, azimuth, distance)
    
    # 提取目标点的纬度和经度
    lat = result['lon2']
    lon = result['lat2']
    
    # 计算目标点的高度
    h = h_ref + up
    
    return [lat, lon, h]

def haversine_distance(point1, point2):
    """
    计算两点之间的距离（以米为单位），基于经纬度坐标。
    
    Args:
        lat1, lon1: 第一个点的纬度（latitude）和经度（longitude），单位：度
        lat2, lon2: 第二个点的纬度（latitude）和经度（longitude），单位：度
    
    Returns:
        distance: 两点之间的距离，单位：米
    """
    # 地球平均半径（米）
    R = 6371000
    lat1 = point1[0]
    lon1 = point1[1]
    lat2 = point2[0]
    lon2 = point2[1]
    
    # 将经纬度从度转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # 纬度和经度的差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine 公式
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # 计算距离
    distance = R * c
    
    return distance

def save_segment_shapefile_with_numbers(
        lane_segments, generated_type, segment_id, lane_count, output_prefix, *,
        point_transform=None, output_crs='EPSG:4326', distance_function=None,
        return_frame=False):
    """Export lanes, or return a frame without writing when return_frame is true.

    point_transform accepts one XYZ point. distance_function accepts two
    transformed XY points and returns metres. Defaults retain legacy conversion.
    Frame mode propagates errors; legacy file mode retains its boolean result.
    """
    try:
        # 创建车道线的GeoDataFrame
        lane_geometries = []
        lane_types = []
        lane_ids = []
        left_lanes = []
        right_lanes = []
        print('lane_segments', lane_segments)
        
        lat_ref, lon_ref, h_ref = 32.022252, 118.726685, 8.135000
        transform_point = point_transform if point_transform is not None else (
            lambda point: enu_to_wgs84(lat_ref, lon_ref, h_ref, point))
        transformed_distance = distance_function if distance_function is not None else haversine_distance

        if generated_type == 0:
            # 按车道类型分组处理
            lane_groups = {}
            for i, (segment, lane_type, lane_num) in enumerate(lane_segments):
                # Keep each feature's metadata with its complete window sequence.
                lane_groups[(lane_type, lane_num, i)] = [segment]
                
            # 处理每种车道类型
            for (lane_type, lane_num, lane_id), segments in lane_groups.items():
                if lane_type == 'left solid':  # 左右车道线类型，需要连接相邻segment
                    print('left solid')
                    # 收集所有线段点
                    all_line_coords = []
                    # print('segments', segments)
                    pre_segment = None
                    i = 0
                    for segment in segments[0]:
                        for line_segment in segment:
                            # 每个line_segment是(2,3)的数组，包含起点和终点
                            if type(line_segment) == np.ndarray:
                                segment_WGS = transform_point(line_segment.tolist())
                                # all_line_coords.append(segment_WGS[:2])
                            else:
                                segment_WGS = transform_point(line_segment)
                                # all_line_coords.append(segment_WGS[:2])
                            if i == 0:
                                pre_segment = segment_WGS[:2]
                                i+=1
                                continue

                            segment_concate = []
                            current_segment = segment_WGS[:2]
                            segment_concate.append(pre_segment)
                            segment_concate.append(current_segment)
                            # print('segment_concate', segment_concate)
                            line = LineString(segment_concate)
                            
                            lane_geometries.append(line)
                            lane_types.append(lane_type)
                            lane_ids.append(lane_id)
                            left_lanes.append(lane_num + 1)  # 左侧车道编号
                            right_lanes.append(None)  # 右侧车道编号为空

                            pre_segment = current_segment                            
                            i+=1
                    
                    # 创建连接的LineString
                    # if all_line_coords:
                    #     line = LineString(all_line_coords)
                    #     lane_geometries.append(line)
                    #     lane_types.append(lane_type)
                    #     lane_ids.append(segments[0][2])  # 使用第一个segment的ID
                    #     left_lanes.append(segments[0][1] + 1)  # 左侧车道编号
                    #     right_lanes.append(None)  # 右侧车道编号为空
                elif lane_type == 'right solid':  # 右车道线类型，需要连接相邻segment
                    print('right solid')
                    # 收集所有线段点
                    all_line_coords = []
                    pre_segment = None
                    i = 0
                    for segment in segments[0]:
                        
                        for line_segment in segment:
                            # print('line_segment', line_segment)
                            # 每个line_segment是(2,3)的数组，包含起点和终点
                            if type(line_segment) == np.ndarray:
                                segment_WGS = transform_point(line_segment.tolist())
                                # all_line_coords.append(segment_WGS[:2])
                            else:
                                segment_WGS = transform_point(line_segment)
                                # all_line_coords.append(segment_WGS[:2])
                            # all_line_coords.extend(line_segment[:, :2].tolist())
                            if i==0:
                                pre_segment = segment_WGS[:2]
                                i+=1
                                continue

                            segment_concate = []
                            current_segment = segment_WGS[:2]
                            segment_concate.append(pre_segment)
                            segment_concate.append(current_segment)
                            # print('segment_concate', segment_concate)
                            line = LineString(segment_concate)
                            lane_geometries.append(line)
                            lane_types.append(lane_type)
                            lane_ids.append(lane_id)
                            left_lanes.append(lane_num + 1)  # 左侧车道编号
                            right_lanes.append(None)  # 右侧车道编号为空

                            pre_segment = current_segment 
                            i+=1
                    
                    # 创建连接的LineString
                    # if all_line_coords:
                    #     line = LineString(all_line_coords)
                    #     lane_geometries.append(line)
                    #     lane_types.append(lane_type)
                    #     lane_ids.append(segments[0][2])  # 使用第一个segment的ID
                    #     left_lanes.append(None)  # 左侧车道编号为空
                    #     right_lanes.append(segments[0][1])     # 右侧车道编号

                else:  # 其他类型（如dashed），保持原有处理方式
                    print('lane type', lane_type)
                    # 收集所有线段点，检测断开的segment
                    all_line_coords_list = []  # 存储多个连续的线段点集合
                    current_line_coords = []
                    for dash_segments in segments:
                        left_dash_segments = []
                        right_dash_segments = []
                        # print('dash_segments {} '.format(dash_segments))
                        for segment_idx, segment in enumerate(dash_segments):
                            # print('segment_idx {}, shape {}'.format(segment_idx, np.asarray(segment).shape))
                            # print('segment {}'.format(segment))
                            if segment_idx == 0:
                                # 第一个segment，直接添加
                                for line_segment in segment:
                                    # print('3333333333 line_segment', line_segment)
                                    if type(line_segment) == np.ndarray:
                                        segment_WGS = transform_point(line_segment.tolist())
                                        current_line_coords.append(segment_WGS[:2])
                                    else:
                                        segment_WGS = transform_point(line_segment)
                                        current_line_coords.append(segment_WGS[:2])
                            else:
                                # 这里出现左右两条虚线，每条虚线分别建立一条线
                                if len(np.asarray(segment).shape) > 2:
                                    
                                    for l, dash_line_segments in enumerate(segment):
                                        if l==0:
                                            for line_segment in segment[l]:
                                                if type(line_segment) == np.ndarray:
                                                    segment_WGS = transform_point(line_segment.tolist())
                                                    left_dash_segments.append(segment_WGS[:2])
                                                else:
                                                    segment_WGS = transform_point(line_segment)
                                                    left_dash_segments.append(segment_WGS[:2])
                                        else:
                                            for line_segment in segment[l]:
                                                if type(line_segment) == np.ndarray:
                                                    segment_WGS = transform_point(line_segment.tolist())
                                                    right_dash_segments.append(segment_WGS[:2])
                                                else:
                                                    segment_WGS = transform_point(line_segment)
                                                    right_dash_segments.append(segment_WGS[:2])
                                    # print('left_dash_segments', left_dash_segments)
                                    # print('right_dash_segments', right_dash_segments)

                                else:
                                    if len(np.asarray(segment).shape) > 2:
                                        prev_segment = dash_segments[segment_idx - 1]
                                    else:
                                        prev_segment = dash_segments[segment_idx - 1]
                                
                                    # print('-1111111111 prev_segment', prev_segment)
                                    # print('-1111111111 segment', segment)
                                    # print('-2222222222 segment shape', np.asarray(segment).shape)
                                    # print('-3333333333 len shape', len(np.asarray(segment).shape))
                                
                                    # 获取前一个segment的最后一个line_segment的终点
                                    prev_last_line_segment = prev_segment[-1]  # 前一个segment的最后一个line_segment
                                    prev_end_point = prev_last_line_segment[:2]  # 前一个segment的终点
                                
                                    # 获取当前segment的第一个line_segment的起点
                                    if len(np.asarray(segment).shape) > 2:
                                        curr_first_line_segment = segment[0]  # 当前segment的第一个line_segment
                                        curr_start_point = curr_first_line_segment[0][:2]   # 当前segment的起点
                                    else:
                                        curr_first_line_segment = segment[0]
                                        curr_start_point = curr_first_line_segment[:2]   # 当前segment的起点
                                
                                    # print('-prev_end_point', prev_end_point)
                                    # print('-curr_start_point', curr_start_point)
                                
                                    # 计算距离
                                    distance = np.linalg.norm(np.array(curr_start_point) - np.array(prev_end_point))
                                    print(f'segment {segment_idx-1} 到 {segment_idx} 的距离: {distance:.2f}m')
                                
                                    if distance > 60:  # 如果距离大于60米，说明断开了
                                        # 保存当前收集的线段点
                                        if current_line_coords:
                                            all_line_coords_list.append(current_line_coords)
                                        # 重新开始收集
                                        current_line_coords = []
                                        print(f"-检测到segment断开，距离: {distance:.2f}m，重新开始收集")
                                
                                    # 添加当前segment的点
                                    for line_segment in segment:
                                        if type(line_segment) == np.ndarray:
                                            segment_WGS = transform_point(line_segment.tolist())
                                            current_line_coords.append(segment_WGS[:2])
                                        else:
                                            segment_WGS = transform_point(line_segment)
                                            current_line_coords.append(segment_WGS[:2])

                                print("========================================")
                        
                        single_end_point = current_line_coords[-1]
                        # print('single_end_point', single_end_point)

                        if len(left_dash_segments) > 0:
                            double_start_point = left_dash_segments[0]
                            distance = transformed_distance(double_start_point, single_end_point)
                            print('检测到距离为 distance', distance)
                            if distance > 60:
                                all_line_coords_list.append(left_dash_segments)
                        
                                if len(right_dash_segments) > 0:
                                    all_line_coords_list.append(right_dash_segments)
                            else:
                                left_dash_segments.insert(0, single_end_point)
                                all_line_coords_list.append(left_dash_segments)
                                right_dash_line = current_line_coords+right_dash_segments
                                all_line_coords_list.append(right_dash_line)
                    
                    # 添加最后一个线段点集合
                    if current_line_coords:
                        all_line_coords_list.append(current_line_coords)
                    
                    # print('all_line_coords_list', all_line_coords_list)
                    # 为每个连续的线段点集合创建LineString
                    for line_coords in all_line_coords_list:
                        if len(line_coords) >= 2:  # 至少需要2个点才能形成线段
                            line = LineString(line_coords)
                            lane_geometries.append(line)
                            lane_types.append(lane_type)
                            lane_ids.append(lane_id)
                            left_lanes.append(lane_num + 1)  # 左侧车道编号
                            right_lanes.append(lane_num)     # 右侧车道编号
        
        elif generated_type == 1:
            dealed_id = []
            
            for i, (segments, lane_type, lane_idx) in enumerate(lane_segments):
                if lane_type == 'solid' and lane_idx not in dealed_id:
                    dealed_id.append(lane_idx)
                    pre_segment = segments[0]
                    # print('pre_segment', pre_segment)
                    for j, segment in enumerate(segments):
                        # print('segment', segment)
                        if j == 0:
                            pre_segment = segment
                            continue
                        
                        # 计算pre_segment与当前segment起点之间的距离，小于0.5m则跳过
                        p0 = np.asarray(pre_segment[:2], dtype=float)
                        p1 = np.asarray(segment[:2], dtype=float)
                        if j == 1 and np.linalg.norm(p1 - p0) < 1:
                            pre_segment = segment
                            continue
                        elif j == len(segments) - 1 and np.linalg.norm(p1 - p0) < 1:
                            # pre_segment = segment
                            continue
                        # 仅使用XY坐标构造两点折线
                        segment_concate = []
                        
                        pre_segment_WGS = transform_point(pre_segment)
                        segment_WGS = transform_point(segment)
                        # print('-------------pre_segment', pre_segment)
                        segment_concate.append(pre_segment_WGS[:2])
                        segment_concate.append(segment_WGS[:2])

                        # segment_concate.append(pre_segment[:2].tolist())
                        # segment_concate.append(segment[:2].tolist())
                        # print('segment_concate', segment_concate)

                        line = LineString(segment_concate)
                        lane_geometries.append(line)
                        lane_types.append(lane_type)
                        lane_ids.append(i)
                        left_lanes.append(None)  # 左侧车道编号为空
                        right_lanes.append(None)     # 右侧车道编号
                        pre_segment = segment
                else:
                    dealed_id.append(lane_idx)
                    pre_segment = segments[0]
                    
                    for j, segment in enumerate(segments):
                        if j == 0:
                            pre_segment = segment
                            continue
                        p0 = np.asarray(pre_segment[:2], dtype=float)
                        p1 = np.asarray(segment[:2], dtype=float)
                        if j == 1 and np.linalg.norm(p1 - p0) < 1:
                            pre_segment = segment
                            continue
                        elif j == len(segments) - 1 and np.linalg.norm(p1 - p0) < 1:
                            # pre_segment = segment
                            continue

                        segment_concate = []
                        pre_segment_WGS = transform_point(pre_segment)
                        segment_WGS = transform_point(segment)
                        segment_concate.append(pre_segment_WGS[:2])
                        segment_concate.append(segment_WGS[:2])

                        # segment_concate.append(pre_segment[:2].tolist())
                        # segment_concate.append(segment[:2].tolist())

                        # print('segment_concate', segment_concate)
                        line = LineString(segment_concate)  # 只使用XY坐标
                        lane_geometries.append(line)
                        lane_types.append(lane_type)
                        lane_ids.append(i)
                        left_lanes.append(None)  # 左侧车道编号为空
                        right_lanes.append(None)     # 右侧车道编号
                        pre_segment = segment
                    # for line_segment in segment:
                    # line = LineString(segment[:, :2])  # 只使用XY坐标
                    # lane_geometries.append(line)
                    # lane_types.append(lane_type)
                    # lane_ids.append(i)
                    # left_lanes.append(None)  # 左侧车道编号为空
                    # right_lanes.append(None)     # 右侧车道编号

        lane_gdf = gpd.GeoDataFrame({
            'id': lane_ids,
            'type': lane_types,
            'segment_id': [segment_id] * len(lane_geometries),
            'left_lane': left_lanes,
            'right_lane': right_lanes,
            'geometry': lane_geometries
        }, geometry='geometry', crs=output_crs)
        # Consecutive windows can repeat their shared endpoint.
        nonzero = [line.length != 0 for line in lane_geometries]
        lane_gdf = lane_gdf.loc[nonzero].reset_index(drop=True)
        if return_frame:
            return lane_gdf

        if not lane_gdf.empty:
            
            # 保存车道线Shapefile
            lane_shp_file = f"shp_temp/{output_prefix}_segment_{segment_id:03d}_lanes.shp"
            lane_gdf.to_file(lane_shp_file)
            print(f"段 {segment_id} 车道线已保存至: {lane_shp_file}")

        return True
    except Exception as e:
        if return_frame:
            raise
        print(f"保存段 {segment_id} Shapefile失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="处理车道线段数据并生成矢量化地图")
    parser.add_argument("-i", "--input_dir", type=str, default="segments_arrow", help="输入目录，包含车道线段数据")
    # parser.add_argument("-i", "--input_dir", type=str, default="straight_segments", help="输入目录，包含车道线段数据")
    parser.add_argument("-o", "--output_file", type=str, default="vector_map_processed.json", help="输出矢量化地图文件")
    parser.add_argument("--min_displacement", type=float, default=50.0, help="最小位移阈值（米）")
    parser.add_argument("--visualize", action="store_true", help="可视化结果", default=True)
    parser.add_argument("--visualize_3d", action="store_true", help="使用3D可视化", default=True)
    parser.add_argument("--visualize_each", action="store_true", help="单独可视化每个段", default=True)
    parser.add_argument("--shp", action="store_true", help="保存为Shapefile格式", default=True)
    parser.add_argument("--shp_prefix", type=str, default="vector_map", help="Shapefile文件前缀")
    parser.add_argument("--max_connect_distance", type=float, default=5.0, help="最大连接距离（米）")
    parser.add_argument("--max_angle_diff", type=float, default=30.0, help="最大角度差异（度）")
    args = parser.parse_args()
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        return
    
    # 查找所有车道线段
    segment_files = []
    if args.input_dir == "segments_arrow":
        for file in os.listdir(args.input_dir):
            if file.startswith("lane_segment_") and file.endswith(".pcd"):
                if file.split("_")[2].split(".")[0] == "incomplete" or file.split("_")[2].split(".")[0] != "001":
                    continue
                segment_id = int(file.split("_")[2].split(".")[0])
                segment_files.append(segment_id)

    elif args.input_dir == "straight_segments":
        for file in os.listdir(args.input_dir):
            if file.startswith("lane_segment_") and file.endswith(".pcd"):
                print(file.split("_")[2].split(".")[0])
                if file.split("_")[2].split(".")[0] != "003":
                    continue
                segment_id = int(file.split("_")[2].split(".")[0])
                print(f"segment_id: {segment_id}")
                segment_files.append(segment_id)
    else:
        print("输入目录错误！")
        return
    
    segment_files.sort()
    print(f"找到 {len(segment_files)} 个车道线段")
    
    # 处理每个车道线段
    all_lane_segments = []
    all_stop_line_positions = []
    segment_lane_counts = {}
    
    for segment_id in segment_files:
        print(f"处理车道线段 {segment_id}...")
        lane_segments, generated_type, lane_count = process_lane_segment(
            args.input_dir, 
            segment_id, 
            args.min_displacement,
            visualize=args.visualize_each,
            visualize_3d=args.visualize_3d
        )
        
        if lane_segments is not None:
            # 记录车道数量
            segment_lane_counts[segment_id] = lane_count
            
            # 对车道线进行编号
            # 使用第一个位姿作为参考
            pose_file = os.path.join(args.input_dir, f"lane_segment_{segment_id:03d}_poses.txt")
            poses = load_poses(pose_file)
            if poses:
                reference_pose = poses[0][1]  # 第一个位姿的数据部分
                # 检查lane_segments的格式，如果已经包含编号信息，则直接使用
                if len(lane_segments) > 0 and isinstance(lane_segments[0], tuple) and len(lane_segments[0]) == 3:
                    numbered_lanes = lane_segments
                else:
                    numbered_lanes, _ = identify_and_number_lanes(lane_segments, reference_pose)
                all_lane_segments.extend(numbered_lanes)
            else:
                # 如果没有位姿信息，就不进行编号
                if len(lane_segments) > 0 and isinstance(lane_segments[0], tuple) and len(lane_segments[0]) == 3:
                    all_lane_segments.extend(lane_segments)
                else:
                    all_lane_segments.extend([(seg[0], seg[1], -1) for seg in lane_segments])
            
            # 为每个段保存单独的Shapefile（包含车道编号信息）
            if args.shp and poses:
                save_segment_shapefile_with_numbers(lane_segments, generated_type, segment_id, lane_count, args.shp_prefix)
                # if len(lane_segments) > 0 and isinstance(lane_segments[0], tuple) and len(lane_segments[0]) == 3:
                    # save_segment_shapefile_with_numbers(lane_segments, stop_line_pos, segment_id, lane_count, args.shp_prefix)
                # else:
                    # numbered_lanes, _ = identify_and_number_lanes(lane_segments, reference_pose)
                    # save_segment_shapefile_with_numbers(numbered_lanes, stop_line_pos, segment_id, lane_count, args.shp_prefix)
    
    # 输出每段的车道数量统计
    print("\n车道数量统计:")
    for segment_id, count in segment_lane_counts.items():
        print(f"段 {segment_id}: {count} 条车道")
    
    print(f"处理完成，共有 {len(all_lane_segments)} 条车道线和 {len(all_stop_line_positions)} 个停止线")
    
    # 保存为整体Shapefile
    if args.shp and all_lane_segments:
        # 创建车道线的GeoDataFrame
        lane_geometries = []
        lane_types = []
        lane_ids = []
        left_lanes = []
        right_lanes = []
        
        for i, (segment, lane_type, lane_num) in enumerate(all_lane_segments):
            # 创建LineString几何对象
            # segment是一个包含多个(2,3)数组的列表，每个数组代表一条线段
            # 将每个线段转换为LineString并合并
            line_coords = []
            for line_segment in segment:
                # 每个line_segment是(2,3)的数组，包含起点和终点
                line_coords.extend(line_segment[:, :2].tolist())  # 只使用XY坐标
            
            # 创建LineString几何对象
            line = LineString(line_coords)
            lane_geometries.append(line)
            lane_types.append(lane_type)
            lane_ids.append(i)
            left_lanes.append(lane_num + 1 if lane_num >= 0 else -1)  # 左侧车道编号
            right_lanes.append(lane_num if lane_num >= 0 else -1)     # 右侧车道编号
        
        if lane_geometries:
            lane_gdf = gpd.GeoDataFrame({
                'id': lane_ids,
                'type': lane_types,
                'left_lane': left_lanes,
                'right_lane': right_lanes,
                'geometry': lane_geometries
            }, crs='EPSG:4326')  # 设置WGS84坐标参考系
            
            # 保存车道线Shapefile
            lane_shp_file = f"{args.shp_prefix}_all_lanes.shp"
            lane_gdf.to_file(lane_shp_file)
            print(f"所有车道线已保存至: {lane_shp_file}")
        
        # 创建停止线的GeoDataFrame
        if all_stop_line_positions:
            stop_line_geometries = []
            stop_line_ids = []
            
            for i, pos in enumerate(all_stop_line_positions):
                # 创建Point几何对象
                point = Point(pos[0], pos[1])
                stop_line_geometries.append(point)
                stop_line_ids.append(i)
            
            stop_line_gdf = gpd.GeoDataFrame({
                'id': stop_line_ids,
                'geometry': stop_line_geometries
            }, crs='EPSG:4326')  # 设置WGS84坐标参考系
            
            # 保存停止线Shapefile
            stop_line_shp_file = f"{args.shp_prefix}_all_stop_lines.shp"
            stop_line_gdf.to_file(stop_line_shp_file)
            print(f"所有停止线已保存至: {stop_line_shp_file}")
    
    # 可视化整体结果
    if args.visualize and all_lane_segments and not args.visualize_each:
        # 提取拟合后的线段点
        fitted_segments = [seg[0] for seg in all_lane_segments]
        
        if args.visualize_3d:
            visualize_lane_segments_3d(all_lane_segments, all_stop_line_positions, title="所有段的车道线和停止线")
        else:
            visualize_lane_segments(all_lane_segments, all_stop_line_positions, title="所有段的车道线和停止线")

# 新增局部拟合专用RANSAC函数

def fit_line_ransac_local(points, max_trials=100, residual_threshold=0.5):
    """
    局部拟合专用RANSAC直线拟合，直接对点集做RANSAC，不做方向约束和过度剔除。
    points: (N,3)
    返回: (2,3)直线端点
    """
    if len(points) < 2:
        return None
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        random_state=42
    )
    try:
        ransac.fit(X, y)
        x_min, x_max = np.min(X), np.max(X)
        x_range = np.array([[x_min], [x_max]])
        y_range = ransac.predict(x_range)
        z_avg = np.mean(points[:, 2])
        line_points = np.column_stack((x_range.flatten(), y_range.flatten(), [z_avg, z_avg]))
        return line_points
    except Exception as e:
        print(f"局部RANSAC拟合失败: {e}")
        return None

def extract_dashed_lane_points(points, poses, dashed_class=2, solid_class=1, vehicle_forward_direction=None, extension_ratio=0.5):
    """
    使用直线拟合虚线点云，并沿首尾方向合并实线点云
    
    Args:
        points: 点云数据 (N, 4) - [x, y, z, class_id]
        dashed_class: 虚线类别ID
        solid_class: 实线类别ID
        extension_ratio: 扩展比例
    """
    # 1. 分离虚线和实线点
    dashed_points = points[points[:, 3] == dashed_class]
    solid_points = points[points[:, 3] == solid_class]
    if len(dashed_points) < 10:
        return dashed_points

    # 3. 利用虚线点云分布计算主方向
    coords_3d = dashed_points[:, :3]
    
    # 计算质心
    centroid = np.mean(coords_3d, axis=0)
    
    # 计算协方差矩阵
    centered_coords = coords_3d - centroid
    cov_matrix = np.cov(centered_coords.T)
    
    # 计算主方向（最大特征值对应的特征向量）
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_dir = eigenvectors[:, np.argmax(eigenvalues)]
    
    print(f"计算得到的主方向: {principal_dir}")
    
    # 4. 用robust_fit_line拟合直线段
    coords_3d = dashed_points[:, :3]
    line_result = fit_line_3d_ransac(coords_3d)
    if line_result[0] is None:
        print("robust_fit_line拟合失败，返回原始虚线点")
        return dashed_points
    
    line_points, inlier_mask = line_result
    start_point, end_point = line_points[0], line_points[1]
    line_direction = end_point - start_point
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    print(f"拟合直线方向: {line_direction}")
    
    # 评估虚线点云的非线性离散程度
    nonlinearity_score = evaluate_dashed_lane_nonlinearity(dashed_points, line_points, line_direction)
    print(f"虚线点云非线性离散程度评分: {nonlinearity_score:.4f}")
    # if nonlinearity_score>17:
        # return dashed_points, nonlinearity_score
    
    # 将poses转换为numpy数组格式
    poses_array = np.array([pose[1] for pose in poses])  # 提取位姿数据部分
    direction_change_std = evaluate_vehicle_trajectory_nonlinearity(poses_array)
    if direction_change_std > 1.2 or nonlinearity_score>17:  # 如果轨迹非线性程度很高
        print("检测到高度非线性")
        return dashed_points, nonlinearity_score, direction_change_std

    # 5. 
    lateral_thresh = 0.8  # 横向距离阈值（米）
    
    # 计算所有实线点到虚线直线段的横向距离
    candidates = []
    for pt in solid_points:
        # 计算点到直线的距离
        vec_to_start = pt[:3] - start_point
        proj_length = np.dot(vec_to_start, line_direction)
        
        # 计算横向距离
        proj_point = start_point + proj_length * line_direction
        lateral_dist = np.linalg.norm(pt[:3] - proj_point)
        
        # 如果横向距离在阈值内，则加入候选
        if lateral_dist < lateral_thresh:
            candidates.append(pt)

    # 6. 将符合条件的实线点云重新归类为虚线类别
    if candidates:
        candidates = np.array(candidates)
        # 将实线点云的类别ID改为虚线类别
        candidates[:, 3] = dashed_class
        print(f"将 {len(candidates)} 个实线点重新归类为虚线类别")
    # 7. 合并点云
    merged_points = [dashed_points]
    if len(candidates) > 0:
        merged_points.append(candidates)
    
    merged_points = np.vstack(merged_points)
    print(f"符合条件的实线点数: {len(candidates)}")
    print(f"合并后总点数: {len(merged_points)}")
    
    # 8. 可视化拟合结果
    if len(candidates) > 0:
        visualize_dashed_lane_fitting(dashed_points, candidates, line_result[0])
    
    return merged_points, nonlinearity_score, direction_change_std

def expand_fusion_points(dashed_points, solid_points, dashed_class):
    """
    扩展融合点云

    使用直线拟合虚线点云，并沿首尾方向合并实线点云
    
    Args:
        points: 点云数据 (N, 4) - [x, y, z, class_id]
        dashed_class: 虚线类别ID
        solid_class: 实线类别ID
    """
    if len(dashed_points) < 10:
        return dashed_points

    # 3. 利用虚线点云分布计算主方向
    coords_3d = dashed_points[:, :3]
    
    # 计算质心
    centroid = np.mean(coords_3d, axis=0)
    
    # 计算协方差矩阵
    centered_coords = coords_3d - centroid
    cov_matrix = np.cov(centered_coords.T)
    
    # 计算主方向（最大特征值对应的特征向量）
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_dir = eigenvectors[:, np.argmax(eigenvalues)]
    
    print(f"计算得到的主方向: {principal_dir}")
    
    # 4. 用robust_fit_line拟合直线段
    coords_3d = dashed_points[:, :3]
    line_result = fit_line_3d_ransac(coords_3d)
    if line_result[0] is None:
        print("robust_fit_line拟合失败，返回原始虚线点")
        return dashed_points
    
    line_points, inlier_mask = line_result
    start_point, end_point = line_points[0], line_points[1]
    line_direction = end_point - start_point
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    print(f"拟合直线方向: {line_direction}")

    # 5. 
    lateral_thresh = 1.5  # 横向距离阈值（米）
    
    # 计算所有实线点到虚线直线段的横向距离
    candidates = []
    for pt in solid_points:
        # 计算点到直线的距离
        vec_to_start = pt[:3] - start_point
        proj_length = np.dot(vec_to_start, line_direction)
        
        # 计算横向距离
        proj_point = start_point + proj_length * line_direction
        lateral_dist = np.linalg.norm(pt[:3] - proj_point)
        
        # 如果横向距离在阈值内，则加入候选
        if lateral_dist < lateral_thresh:
            candidates.append(pt)

    # 6. 将符合条件的实线点云重新归类为虚线类别
    if candidates:
        candidates = np.array(candidates)
        # 将实线点云的类别ID改为虚线类别
        candidates[:, 3] = dashed_class
        print(f"将 {len(candidates)} 个实线点重新归类为虚线类别")
    # 7. 合并点云
    merged_points = [dashed_points]
    if len(candidates) > 0:
        merged_points.append(candidates)
    
    merged_points = np.vstack(merged_points)
    print(f"符合条件的实线点数: {len(candidates)}")
    print(f"合并后总点数: {len(merged_points)}")
    
    # 8. 可视化拟合结果
    if len(candidates) > 0:
        visualize_dashed_lane_fitting(dashed_points, candidates, line_result[0])
    
    return merged_points

def extract_dashed_lane_points_curve(points, dashed_class=2, solid_class=1, vehicle_forward_direction=None, extension_ratio=1.0):
    """
    RANSAC剔除离群点 + 端点重复加权 + 三次多项式拟合 + 首尾切线拓展 + 可视化
    """
    # 1. 分离虚线和实线点
    dashed_points = points[points[:, 3] == dashed_class]
    solid_points = points[points[:, 3] == solid_class]
    if len(dashed_points) < 10:
        return dashed_points
    
    coords_3d = dashed_points[:, :3]
    line_result = fit_line_3d_ransac(coords_3d)
    if line_result[0] is None:
        print("robust_fit_line拟合失败，返回原始虚线点")
        return dashed_points
    
    line_points, inlier_mask = line_result
    start_point, end_point = line_points[0], line_points[1]
    line_direction = end_point - start_point
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    print(f"拟合直线方向: {line_direction}")

    # 评估虚线点云的非线性离散程度
    nonlinearity_score = evaluate_dashed_lane_nonlinearity(dashed_points, line_points, line_direction)
    print(f"虚线点云非线性离散程度评分: {nonlinearity_score:.4f}")
    if nonlinearity_score>17:
        return dashed_points, nonlinearity_score
    
    coords_3d = dashed_points[:, :3]
    x = coords_3d[:, 0]
    sort_idx = np.argsort(x)
    coords_3d = coords_3d[sort_idx]
    x = x[sort_idx]
    n = len(x)
    n_half = max(3, n // 2)
    # 前半段
    coords_front = coords_3d[:n_half]
    X_front = coords_front[:, 0].reshape(-1, 1)
    y_xy_front = coords_front[:, 1]
    y_xz_front = coords_front[:, 2]
    ransac_xy_front = RANSACRegressor(estimator=LinearRegression(), max_trials=1000, residual_threshold=0.2, random_state=42)
    ransac_xz_front = RANSACRegressor(estimator=LinearRegression(), max_trials=1000, residual_threshold=0.2, random_state=42)
    ransac_xy_front.fit(X_front, y_xy_front)
    ransac_xz_front.fit(X_front, y_xz_front)
    inlier_mask_front = ransac_xy_front.inlier_mask_ & ransac_xz_front.inlier_mask_
    x_front = coords_front[:, 0][inlier_mask_front]
    y_front = coords_front[:, 1][inlier_mask_front]
    z_front = coords_front[:, 2][inlier_mask_front]
    # 后半段
    coords_back = coords_3d[-n_half:]
    X_back = coords_back[:, 0].reshape(-1, 1)
    y_xy_back = coords_back[:, 1]
    y_xz_back = coords_back[:, 2]
    ransac_xy_back = RANSACRegressor(estimator=LinearRegression(), max_trials=1000, residual_threshold=0.2, random_state=42)
    ransac_xz_back = RANSACRegressor(estimator=LinearRegression(), max_trials=1000, residual_threshold=0.2, random_state=42)
    ransac_xy_back.fit(X_back, y_xy_back)
    ransac_xz_back.fit(X_back, y_xz_back)
    inlier_mask_back = ransac_xy_back.inlier_mask_ & ransac_xz_back.inlier_mask_
    x_back = coords_back[:, 0][inlier_mask_back]
    y_back = coords_back[:, 1][inlier_mask_back]
    z_back = coords_back[:, 2][inlier_mask_back]
    # 多项式拟合
    poly_xy_front = np.polyfit(x_front, y_front, 3)
    poly_xz_front = np.polyfit(x_front, z_front, 3)
    p_xy_front = np.poly1d(poly_xy_front)
    p_xz_front = np.poly1d(poly_xz_front)
    dp_xy_front = p_xy_front.deriv()
    dp_xz_front = p_xz_front.deriv()
    poly_xy_back = np.polyfit(x_back, y_back, 3)
    poly_xz_back = np.polyfit(x_back, z_back, 3)
    p_xy_back = np.poly1d(poly_xy_back)
    p_xz_back = np.poly1d(poly_xz_back)
    dp_xy_back = p_xy_back.deriv()
    dp_xz_back = p_xz_back.deriv()
    # 首尾点
    x_min, x_max = np.min(x), np.max(x)
    start_point = np.array([x_min, p_xy_front(x_min), p_xz_front(x_min)])
    end_point = np.array([x_max, p_xy_back(x_max), p_xz_back(x_max)])
    # 首尾切线
    start_tangent = np.array([1, dp_xy_front(x_min), dp_xz_front(x_min)])
    start_tangent = start_tangent / np.linalg.norm(start_tangent)
    end_tangent = np.array([1, dp_xy_back(x_max), dp_xz_back(x_max)])
    end_tangent = end_tangent / np.linalg.norm(end_tangent)
    # 其余拓展和可视化逻辑保持不变

    # 6. 拓展区域参数
    extension_dist = (x_max - x_min) * extension_ratio
    candidates = []
    for pt in solid_points:
        # 判断点是否在首端延长线上
        vec_start = pt[:3] - start_point
        proj_len_start = np.dot(vec_start, start_tangent)
        lateral_dist_start = np.linalg.norm(vec_start - proj_len_start * start_tangent)
        if 0 < proj_len_start < extension_dist and lateral_dist_start < 1.0:
            candidates.append(pt)
            continue
        # 判断点是否在尾端延长线上
        vec_end = pt[:3] - end_point
        proj_len_end = np.dot(vec_end, end_tangent)
        lateral_dist_end = np.linalg.norm(vec_end - proj_len_end * end_tangent)
        if 0 < proj_len_end < extension_dist and lateral_dist_end < 1.0:
            candidates.append(pt)

    # 7. 合并点云
    if candidates:
        candidates = np.array(candidates)
        candidates[:, 3] = dashed_class
        merged_points = np.vstack([dashed_points, candidates])
    else:
        merged_points = dashed_points

    # 8. 可视化
    # try:
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name="多项式拟合与拓展可视化", width=1400, height=900)
    #     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    #     vis.add_geometry(coordinate_frame)
    #     # 原始虚线点云（红色）
    #     pcd_dashed = o3d.geometry.PointCloud()
    #     pcd_dashed.points = o3d.utility.Vector3dVector(dashed_points[:, :3])
    #     pcd_dashed.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(dashed_points)))
    #     vis.add_geometry(pcd_dashed)
    #     # 拟合曲线（蓝色线）
    #     x_curve_front = np.linspace(x_min, x[int(len(x)/2)], 50)
    #     y_curve_front = p_xy_front(x_curve_front)
    #     z_curve_front = p_xz_front(x_curve_front)
    #     curve_points_front = np.stack([x_curve_front, y_curve_front, z_curve_front], axis=1)
    #     curve_line_front = o3d.geometry.LineSet()
    #     curve_line_front.points = o3d.utility.Vector3dVector(curve_points_front)
    #     curve_line_front.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(curve_points_front)-1)])
    #     curve_line_front.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * (len(curve_points_front)-1)))
    #     vis.add_geometry(curve_line_front)
    #     # 可视化后半段拟合曲线
    #     x_curve_back = np.linspace(x[int(len(x)/2)], x_max, 50)
    #     y_curve_back = p_xy_back(x_curve_back)
    #     z_curve_back = p_xz_back(x_curve_back)
    #     curve_points_back = np.stack([x_curve_back, y_curve_back, z_curve_back], axis=1)
    #     curve_line_back = o3d.geometry.LineSet()
    #     curve_line_back.points = o3d.utility.Vector3dVector(curve_points_back)
    #     curve_line_back.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(curve_points_back)-1)])
    #     curve_line_back.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * (len(curve_points_back)-1)))
    #     vis.add_geometry(curve_line_back)
    #     # 拓展得到的实线点云（绿色）
    #     if candidates is not None and len(candidates) > 0:
    #         pcd_candidates = o3d.geometry.PointCloud()
    #         pcd_candidates.points = o3d.utility.Vector3dVector(candidates[:, :3])
    #         pcd_candidates.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]] * len(candidates)))
    #         vis.add_geometry(pcd_candidates)
    #     # 首尾点球体
    #     start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    #     start_sphere.translate(start_point)
    #     start_sphere.paint_uniform_color([1, 0.5, 0])  # 橙色
    #     vis.add_geometry(start_sphere)
    #     end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    #     end_sphere.translate(end_point)
    #     end_sphere.paint_uniform_color([0, 1, 1])  # 青色
    #     vis.add_geometry(end_sphere)
    #     # 首尾切线箭头
    #     arrow_len = extension_dist * 0.8
    #     arrow_start = o3d.geometry.LineSet()
    #     arrow_start.points = o3d.utility.Vector3dVector([start_point, start_point + start_tangent * arrow_len])
    #     arrow_start.lines = o3d.utility.Vector2iVector([[0, 1]])
    #     arrow_start.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])
    #     vis.add_geometry(arrow_start)
    #     arrow_end = o3d.geometry.LineSet()
    #     arrow_end.points = o3d.utility.Vector3dVector([end_point, end_point + end_tangent * arrow_len])
    #     arrow_end.lines = o3d.utility.Vector2iVector([[0, 1]])
    #     arrow_end.colors = o3d.utility.Vector3dVector([[0, 1, 1]])
    #     vis.add_geometry(arrow_end)
    #     # 渲染设置
    #     opt = vis.get_render_option()
    #     opt.background_color = np.array([0.1, 0.1, 0.1])
    #     opt.point_size = 3.0
    #     opt.line_width = 3.0
    #     # 打印拟合信息
    #     print("\n=== 多项式拟合与拓展信息 ===")
    #     print(f"原始虚线点数: {len(dashed_points)}")
    #     # print(f"RANSAC内点数: {len(inlier_points)}")
    #     print(f"拓展实线点数: {len(candidates)}")
    #     print(f"合并后总点数: {len(merged_points)}")
    #     print(f"首点坐标: {start_point}")
    #     print(f"尾点坐标: {end_point}")
    #     print(f"首端切线方向: {start_tangent}")
    #     print(f"尾端切线方向: {end_tangent}")
    #     print(f"拓展距离: {extension_dist:.2f} 米")
    #     print("\n=== 可视化图例 ===")
    #     print("红色点: 原始虚线点云")
    #     print("蓝色线: 拟合的三次多项式曲线")
    #     print("绿色点: 拓展得到的实线点云")
    #     print("橙色球+箭头: 首点及首端切线方向")
    #     print("青色球+箭头: 尾点及尾端切线方向")
    #     print("按 'Q' 退出可视化")
    #     vis.run()
    #     vis.destroy_window()
    # except Exception as e:
    #     print(f"可视化失败: {e}")
    return merged_points, nonlinearity_score

def classify_dashed_lane_with_ends(points, poses, vehicle_forward_direction, 
                                  dashed_class=2, solid_class=1,
                                  min_dashed_points=30, min_end_ratio=0.1):
    """
    智能分类虚线车道线，包括两端的实线段
    
    Args:
        points: 所有点云 (N, 4)
        vehicle_forward_direction: 车辆前进方向向量 (3,)
        dashed_class: 虚线类别ID
        solid_class: 实线类别ID
        min_dashed_points: 最小虚线点数
        min_end_ratio: 端部实线段最小比例（相对于总点数）
    
    Returns:
        classified_points: 重新分类后的点云，端部实线段被标记为虚线类别
    """
    # 1. 提取原始虚线点云
    dashed_mask = points[:, 3] == dashed_class
    original_dashed_points = points[dashed_mask]
    
    if len(original_dashed_points) < min_dashed_points:
        return points, {'extended': False, 'reason': '虚线点数不足'}
    
    # 2. 使用扩展函数提取包含端部的虚线点云
    extended_dashed_points, nonlinearity_score, direction_change_std = extract_dashed_lane_points(
        points, poses, dashed_class, solid_class, vehicle_forward_direction)
    
    return extended_dashed_points, nonlinearity_score, direction_change_std

def evaluate_dashed_lane_nonlinearity(dashed_points, line_points, line_direction):
    """
    评估虚线点云相对于拟合直线段的非线性离散程度
    
    Args:
        dashed_points: 虚线点云 (N, 4)
        line_points: 拟合的直线段 (2, 3)
        line_direction: 直线方向向量 (3,)
    
    Returns:
        nonlinearity_score: 非线性离散程度评分 (0-1，0表示完全线性，1表示高度非线性)
    """
    if len(dashed_points) < 3:
        return 0.0
    
    coords_3d = dashed_points[:, :3]
    start_point, end_point = line_points[0], line_points[1]
    
    # 1. 计算所有点到直线的距离
    distances = []
    for point in coords_3d:
        # 计算点到直线的垂直距离
        vec_to_start = point - start_point
        proj_length = np.dot(vec_to_start, line_direction)
        proj_point = start_point + proj_length * line_direction
        distance = np.linalg.norm(point - proj_point)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # 2. 计算距离的统计特征
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    max_distance = np.max(distances)
    
    # 3. 计算非线性评分（基于距离的变异系数和最大偏差）
    if mean_distance > 0:
        cv_distance = std_distance / mean_distance  # 变异系数
    else:
        cv_distance = 0
    
    # 4. 计算点云沿直线方向的分布特征
    # 将点投影到直线方向上
    proj_lengths = []
    for point in coords_3d:
        vec_to_start = point - start_point
        proj_length = np.dot(vec_to_start, line_direction)
        proj_lengths.append(proj_length)
    
    proj_lengths = np.array(proj_lengths)
    
    # 计算投影长度的分布特征
    proj_std = np.std(proj_lengths)
    proj_range = np.max(proj_lengths) - np.min(proj_lengths)
    
    # 5. 计算非线性评分
    # 综合考虑距离变异、最大偏差和投影分布
    distance_score = min(1.0, cv_distance * 2)  # 距离变异评分
    max_deviation_score = min(1.0, max_distance / 2.0)  # 最大偏差评分
    distribution_score = min(1.0, proj_std / (proj_range + 1e-6))  # 分布均匀性评分
    
    # 综合评分
    nonlinearity_score = (mean_distance * 0.4 + 
                         max_deviation_score * 0.4 + 
                         proj_std * 0.2)
    
    # 打印详细评估信息
    # print(f"  - 平均距离: {mean_distance:.3f} 米")
    # print(f"  - 距离标准差: {std_distance:.3f} 米")
    # print(f"  - 最大距离: {max_distance:.3f} 米")
    # print(f"  - 距离变异系数: {cv_distance:.3f}")
    # print(f"  - 投影长度标准差: {proj_std:.3f} 米")
    # print(f"  - 投影长度范围: {proj_range:.3f} 米")
    
    return nonlinearity_score

def visualize_dashed_lane_fitting(dashed_points, candidate_points, line_result):
    """
    可视化虚线拟合结果
    
    Args:
        dashed_points: 原始虚线点云 (N, 4)
        candidate_points: 候选实线点云 (M, 4)
        line_points: 拟合的直线段 (2, 3)
        start_point: 直线起点 (3,)
        end_point: 直线终点 (3,)
        line_direction: 直线方向向量 (3,)
    """
    import open3d as o3d
    
    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="虚线拟合可视化", width=1400, height=900)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # 1. 原始虚线点云（红色）
    pcd_dashed = o3d.geometry.PointCloud()
    pcd_dashed.points = o3d.utility.Vector3dVector(dashed_points[:, :3])
    pcd_dashed.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(dashed_points))]))
    vis.add_geometry(pcd_dashed)
    
    # 2. 候选实线点云（绿色）
    if len(candidate_points) > 0:
        pcd_candidates = o3d.geometry.PointCloud()
        pcd_candidates.points = o3d.utility.Vector3dVector(candidate_points[:, :3])
        pcd_candidates.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for _ in range(len(candidate_points))]))
        vis.add_geometry(pcd_candidates)
    
    
    if line_result is not None:
        if line_result.shape[1] == 2:
            z_coords = np.zeros((line_result.shape[0], 1))
            global_3d = np.hstack(line_result, z_coords)
        else:
            global_3d = line_result[:, :3]
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(global_3d)
        lines = [[0, 1]]
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # 黄色
        colors = np.array([[1, 1, 0]])
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 3.0
    opt.line_width = 3.0
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # 打印图例信息
    # print("\n=== 虚线拟合可视化图例 ===")
    # print("红色点: 原始虚线点云")
    # print("绿色点: 候选实线点云（横向距离 < 1.0米）")
    # print("蓝色线: 拟合的直线段")
    # print("橙色箭头: 直线方向")
    # print("紫色球: 直线起点")
    # print("青色球: 直线终点")
    # print("按 'Q' 退出可视化")
    
    # # 打印拟合信息
    # print(f"\n=== 拟合信息 ===")
    # print(f"原始虚线点数: {len(dashed_points)}")
    # print(f"候选实线点数: {len(candidate_points)}")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def evaluate_vehicle_trajectory_nonlinearity(poses, window_size=10, min_poses=20):
    """
    使用位姿数据评估车辆整体轨迹的非线性
    
    Args:
        poses: 车辆位姿数据 (N, 7) - [x, y, z, qw, qx, qy, qz] 或 (N, 6) - [x, y, z, roll, pitch, yaw]
        window_size: 滑动窗口大小，用于局部曲率计算
        min_poses: 最小位姿数量要求
    
    Returns:
        trajectory_info: 包含各种非线性评估指标的字典
    """
    print(f"位姿点数: {len(poses)}")
    if len(poses) < min_poses:
        print(f"位姿点数: {len(poses)}")
        print(f"位姿数量不足: {len(poses)} < {min_poses}")
        return None
    
    # 提取位置信息
    positions = poses[:, :3]  # (N, 3)
    
    # 1. 计算轨迹总长度和直线距离
    total_length = 0.0
    for i in range(1, len(positions)):
        segment_length = np.linalg.norm(positions[i] - positions[i-1])
        total_length += segment_length
    
    straight_distance = np.linalg.norm(positions[-1] - positions[0])
    path_efficiency = straight_distance / (total_length + 1e-6)  # 路径效率
    
    # 2. 计算轨迹的曲率变化
    curvatures = []
    for i in range(window_size, len(positions) - window_size):
        # 取窗口内的点
        window_points = positions[i-window_size:i+window_size+1]
        
        # 用二次多项式拟合局部轨迹
        t = np.arange(len(window_points))
        try:
            # 分别拟合x, y, z
            poly_x = np.polyfit(t, window_points[:, 0], 2)
            poly_y = np.polyfit(t, window_points[:, 1], 2)
            poly_z = np.polyfit(t, window_points[:, 2], 2)
            
            # 计算曲率（简化版本）
            # 对于二次曲线，曲率与二次项系数相关
            curvature = np.sqrt(poly_x[0]**2 + poly_y[0]**2 + poly_z[0]**2)
            curvatures.append(curvature)
        except:
            curvatures.append(0.0)
    
    if len(curvatures) > 0:
        mean_curvature = np.mean(curvatures)
        std_curvature = np.std(curvatures)
        max_curvature = np.max(curvatures)
    else:
        mean_curvature = std_curvature = max_curvature = 0.0
    
    # 3. 计算方向变化
    direction_changes = []
    for i in range(1, len(positions)):
        if i < len(positions) - 1:
            vec1 = positions[i] - positions[i-1]
            vec2 = positions[i+1] - positions[i]
            
            # 归一化
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                vec1_normalized = vec1 / norm1
                vec2_normalized = vec2 / norm2
                
                # 计算夹角
                cos_angle = np.dot(vec1_normalized, vec2_normalized)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                direction_changes.append(np.degrees(angle))
    
    if len(direction_changes) > 0:
        mean_direction_change = np.mean(direction_changes)
        max_direction_change = np.max(direction_changes)
        direction_change_std = np.std(direction_changes)
    else:
        mean_direction_change = max_direction_change = direction_change_std = 0.0
    
    # 3b. 基于弦向的相对初始方向最大角度变化（适合稀疏点，XY平面）
    max_heading_change_rel_init_deg = 0.0
    end_to_start_angle_rel_init_deg = 0.0
    xy_all = positions[:, :2]
    if xy_all.shape[0] >= 2:
        # 累积里程
        diffs = xy_all[1:] - xy_all[:-1]
        seg_dists = np.linalg.norm(diffs, axis=1)
        cum_s = np.zeros(xy_all.shape[0])
        if seg_dists.size > 0:
            cum_s[1:] = np.cumsum(seg_dists)
        # 初始方向：用前5米的弦向
        base_len0 = 5.0
        j0 = None
        for j in range(1, xy_all.shape[0]):
            if cum_s[j] - cum_s[0] >= base_len0:
                j0 = j
                break
        if j0 is None and xy_all.shape[0] >= 2:
            j0 = 1
        if j0 is not None:
            init_vec = xy_all[j0] - xy_all[0]
            init_norm = np.linalg.norm(init_vec)
            if init_norm > 1e-8:
                init_dir = init_vec / init_norm
                # 计算首尾与初始方向夹角
                tail_vec = xy_all[-1] - xy_all[0]
                tail_norm = np.linalg.norm(tail_vec)
                if tail_norm > 1e-8:
                    tail_dir = tail_vec / tail_norm
                    end_to_start_angle_rel_init_deg = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(tail_dir, init_dir)), -1.0, 1.0))))
                base_len = 5.0
                n = xy_all.shape[0]
                for k in range(n):
                    # 找后向点
                    jf = None
                    for j in range(k + 1, n):
                        if cum_s[j] - cum_s[k] >= base_len:
                            jf = j
                            break
                    # 找前向点
                    ib = None
                    for i in range(k - 1, -1, -1):
                        if cum_s[k] - cum_s[i] >= base_len:
                            ib = i
                            break
                    dir_vec = None
                    if ib is not None and jf is not None:
                        dir_vec = xy_all[jf] - xy_all[ib]
                    elif jf is not None:
                        dir_vec = xy_all[jf] - xy_all[k]
                    elif ib is not None:
                        dir_vec = xy_all[k] - xy_all[ib]
                    if dir_vec is None:
                        continue
                    dn = np.linalg.norm(dir_vec)
                    if dn <= 1e-8:
                        continue
                    dir_unit = dir_vec / dn
                    # 绝对点积抑制符号翻转
                    cosv = np.clip(np.abs(np.dot(dir_unit, init_dir)), -1.0, 1.0)
                    ang = np.degrees(np.arccos(cosv))
                    if ang > max_heading_change_rel_init_deg:
                        max_heading_change_rel_init_deg = ang
    
    # 4. 计算轨迹的直线度（用PCA）
    if len(positions) >= 3:
        # 中心化
        centered_positions = positions - np.mean(positions, axis=0)
        
        # PCA分析
        cov_matrix = np.cov(centered_positions.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 主成分解释的方差比例
        total_variance = np.sum(eigenvals)
        if total_variance > 1e-6:
            explained_variance_ratio = eigenvals / total_variance
            linearity_score = explained_variance_ratio[2]  # 最小特征值对应的比例
        else:
            linearity_score = 1.0
    else:
        linearity_score = 1.0
    
    # 5. 计算轨迹的平滑度（相邻点距离的一致性）
    segment_lengths = []
    for i in range(1, len(positions)):
        length = np.linalg.norm(positions[i] - positions[i-1])
        segment_lengths.append(length)
    
    segment_lengths = np.array(segment_lengths)
    length_consistency = 1.0 - (np.std(segment_lengths) / (np.mean(segment_lengths) + 1e-6))
    length_consistency = np.clip(length_consistency, 0.0, 1.0)
    
    # 6. 综合非线性评分
    # 各项指标归一化到0-1范围，1表示最非线性
    curvature_score = min(1.0, mean_curvature * 10)  # 平均曲率评分
    direction_score = min(1.0, mean_direction_change / 30.0)  # 方向变化评分
    efficiency_score = 1.0 - path_efficiency  # 路径效率评分（非线性时效率低）
    smoothness_score = 1.0 - length_consistency  # 平滑度评分
    
    # 综合评分
    nonlinearity_score = (curvature_score * 0.3 + 
                         direction_score * 0.3 + 
                         efficiency_score * 0.2 + 
                         smoothness_score * 0.2)
    
    # 7. 构建返回信息
    trajectory_info = {
        'total_length': total_length,
        'straight_distance': straight_distance,
        'path_efficiency': path_efficiency,
        'mean_curvature': mean_curvature,
        'std_curvature': std_curvature,
        'max_curvature': max_curvature,
        'mean_direction_change': mean_direction_change,
        'max_direction_change': max_direction_change,
        'direction_change_std': direction_change_std,
        'linearity_score': linearity_score,
        'length_consistency': length_consistency,
        'nonlinearity_score': nonlinearity_score,
        'curvature_score': curvature_score,
        'direction_score': direction_score,
        'efficiency_score': efficiency_score,
        'smoothness_score': smoothness_score,
        'max_heading_change_rel_init_deg': max_heading_change_rel_init_deg,
        'end_to_start_angle_rel_init_deg': end_to_start_angle_rel_init_deg
    }
    
    # 8. 打印详细评估信息
    print(f"=== 车辆轨迹非线性评估 ===")
    print(f"最大方向变化: {max_heading_change_rel_init_deg:.2f} 度")
    print(f"首尾与初始方向夹角: {end_to_start_angle_rel_init_deg:.2f} 度")
    # print(f"轨迹总长度: {total_length:.2f} 米")
    # print(f"直线距离: {straight_distance:.2f} 米")
    # print(f"路径效率: {path_efficiency:.3f}")
    # print(f"平均曲率: {mean_curvature:.6f}")
    # print(f"曲率标准差: {std_curvature:.6f}")
    # print(f"最大曲率: {max_curvature:.6f}")
    # print(f"平均方向变化: {mean_direction_change:.2f} 度")
    print(f"最大方向变化: {max_direction_change:.2f} 度")
    print(f"方向变化标准差: {direction_change_std:.2f} 度")
    # print(f"直线度评分: {linearity_score:.3f}")
    # print(f"长度一致性: {length_consistency:.3f}")
    # print(f"综合非线性评分: {nonlinearity_score:.3f}")
    # print(f"  - 曲率评分: {curvature_score:.3f}")
    # print(f"  - 方向评分: {direction_score:.3f}")
    # print(f"  - 效率评分: {efficiency_score:.3f}")
    # print(f"  - 平滑度评分: {smoothness_score:.3f}")
    
    return direction_change_std

def check_and_correct_dashed_line_direction(dashed_line, vehicle_direction, angle_threshold=45.0):
    """
    检查虚线方向与车辆方向的一致性，如果不一致则进行修正（3D版本，标准三维夹角计算）

    参数:
        dashed_line: 拟合的虚线 (2, 3) - 起点和终点
        vehicle_direction: 车辆前进方向 (3,)
        angle_threshold: 角度阈值，超过此角度认为方向不一致（度）

    返回:
        corrected_line: 修正后的虚线 (2, 3)
        direction_consistent: 方向是否一致
        angle_diff: 角度差（度）
    """
    import numpy as np
    if dashed_line is None or len(dashed_line) < 2:
        return dashed_line, False, 0.0
    
    # 保证vehicle_direction为3维
    vehicle_direction = np.array(vehicle_direction)
    if vehicle_direction.shape[0] == 2:
        # 自动补z=0
        vehicle_direction = np.array([vehicle_direction[0], vehicle_direction[1], 0.0])

    def angle_between_vectors_3d(a, b):
        a = np.array(a)
        b = np.array(b)
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        dot_product = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    # 计算虚线方向向量（3D）
    line_vec = dashed_line[1] - dashed_line[0]
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-8:
        return dashed_line, False, 0.0

    # 计算三维夹角
    angle_diff = angle_between_vectors_3d(line_vec, vehicle_direction)

    # 判断方向一致性
    direction_consistent = angle_diff <= angle_threshold
    # print(f"虚线方向与车辆方向一致性: {direction_consistent}")
    if direction_consistent:
        print(f"虚线方向与车辆方向一致，角度差: {angle_diff:.2f}度，保留原始虚线方向")
        return dashed_line, True, angle_diff
    else:
        print(f"虚线方向与车辆方向不一致，角度差: {angle_diff:.2f}度，直接翻转端点")
        # 直接翻转线段端点
        corrected_line = np.array([
            dashed_line[1],
            dashed_line[0]
        ])
        return corrected_line, True, angle_diff

def fit_line_3d_ransac_with_direction_constraint(points, vehicle_direction, max_trials=1000, residual_threshold=0.2):
    """
    使用车辆方向约束的RANSAC三维直线拟合
    
    参数:
        points: 点云数组 (N, 3)
        vehicle_direction: 车辆前进方向 (2,)
        max_trials: RANSAC最大迭代次数
        residual_threshold: 内点阈值（米）
        
    返回:
        line_points: 拟合后的直线点 (2, 3)
        inliers: 内点索引
    """
    if len(points) < 2:
        return None, None
    
    # 归一化车辆方向
    vehicle_dir_norm = vehicle_direction / (np.linalg.norm(vehicle_direction) + 1e-8)
    
    # 计算点云在车辆方向上的投影
    center_point = np.mean(points[:, :2], axis=0)
    projections = np.dot(points[:, :2] - center_point, vehicle_dir_norm)
    
    # 找到投影的最小值和最大值
    min_proj_idx = np.argmin(projections)
    max_proj_idx = np.argmax(projections)
    
    # 创建沿车辆方向的直线
    start_point_2d = center_point + vehicle_dir_norm * projections[min_proj_idx]
    end_point_2d = center_point + vehicle_dir_norm * projections[max_proj_idx]
    
    # 使用对应点的Z坐标
    z_start = points[min_proj_idx, 2]
    z_end = points[max_proj_idx, 2]
    
    # 创建三维直线点
    line_points = np.array([
        [start_point_2d[0], start_point_2d[1], z_start],
        [end_point_2d[0], end_point_2d[1], z_end]
    ])
    
    # 计算内点（距离直线较近的点）
    line_vec = line_points[1] - line_points[0]
    line_length = np.linalg.norm(line_vec)
    if line_length > 1e-8:
        line_direction = line_vec / line_length
        distances = []
        for point in points:
            # 计算点到直线的距离
            point_vec = point - line_points[0]
            projection_length = np.dot(point_vec, line_direction)
            projection_point = line_points[0] + projection_length * line_direction
            distance = np.linalg.norm(point - projection_point)
            distances.append(distance)
        
        inlier_mask = np.array(distances) <= residual_threshold
    else:
        inlier_mask = np.ones(len(points), dtype=bool)
    
    return line_points, inlier_mask

def visualize_single_window(window_points, window_segments, window_center, current_vehicle_dir, window_id, 
                           dashed_class=2, solid_class=1, window_length=20.0, window_width=15.0, window_height=5.0):
    """
    可视化单个窗口的拟合效果
    
    Args:
        window_points: 窗口内的点云
        window_segments: 窗口内拟合的线段 {'dashed': [], 'left_solid': [], 'right_solid': []}
        window_center: 窗口中心点
        current_vehicle_dir: 当前窗口使用的车辆方向
        window_id: 窗口ID
    """
    import open3d as o3d
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"窗口 {window_id} 拟合效果", width=1200, height=800)
    
    # 原始点云（按类别着色）
    coords_3d = window_points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_3d)
    
    colors = np.zeros_like(coords_3d)
    dashed_mask = (window_points[:, 3] == dashed_class)
    solid_mask = (window_points[:, 3] == solid_class)
    colors[dashed_mask] = [1, 0, 0]    # 红色 - 虚线
    colors[solid_mask] = [0, 1, 1]     # 青色 - 实线
    colors[~(dashed_mask | solid_mask)] = [0.7, 0.7, 0.7]  # 灰色 - 其他
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    
    def draw_arraw(segment):
        # 添加端点球体
        for point in segment:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            vis.add_geometry(sphere)
        
        # 添加虚线方向箭头
        # 计算虚线方向
        # 计算虚线方向向量
        line_vec = segment[1] - segment[0]
        line_length = np.linalg.norm(line_vec)
        if line_length > 1e-8:
            line_direction = line_vec / line_length
            
            # 计算箭头位置（在线段中点）
            arrow_position = (segment[0] + segment[1]) / 2
            
            # 箭头参数
            arrow_length = 2.0  # 箭头长度
            arrow_head_length = 0.8  # 箭头头部长度
            arrow_head_width = 0.6   # 箭头头部宽度
            
            # 计算箭头终点
            arrow_end = arrow_position + line_direction * arrow_length
            
            # 创建箭头线段
            arrow_line = o3d.geometry.LineSet()
            arrow_line.points = o3d.utility.Vector3dVector([arrow_position, arrow_end])
            arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            arrow_line.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])  # 橙色
            vis.add_geometry(arrow_line)
            
            # 计算垂直于箭头方向的向量
            perp_dir = np.array([-line_direction[1], line_direction[0], 0])
            
            # 箭头头部的三个顶点
            arrow_tip = arrow_end
            arrow_base = arrow_end - line_direction * arrow_head_length
            
            # 箭头头部的左右两个顶点
            arrow_left = arrow_base + perp_dir * arrow_head_width
            arrow_right = arrow_base - perp_dir * arrow_head_width
            
            # 创建箭头头部的三角形
            arrow_head = o3d.geometry.TriangleMesh()
            arrow_head.vertices = o3d.utility.Vector3dVector([arrow_tip, arrow_left, arrow_right])
            arrow_head.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
            arrow_head.paint_uniform_color([1, 0.5, 0])  # 橙色
            vis.add_geometry(arrow_head)
    # 绘制拟合的线段
    # 虚线线段
    for segment in window_segments.get('dashed', []):
        if segment is not None and len(segment) >= 2:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(segment)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
            vis.add_geometry(line_set)
            draw_arraw(segment)
            
        else:
            print(f"窗口 {window_id}: 没有虚线线段")

    # 左实线线段
    for segment in window_segments.get('left_solid', []):
        if segment is not None and len(segment) >= 2:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(segment)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # 蓝色
            vis.add_geometry(line_set)
            draw_arraw(segment)
            
            # 添加端点球体
            for point in segment:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                sphere.translate(point)
                sphere.paint_uniform_color([0, 0, 1])  # 蓝色
                vis.add_geometry(sphere)
    
    # 右实线线段
    for segment in window_segments.get('right_solid', []):
        if segment is not None and len(segment) >= 2:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(segment)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 绿色
            vis.add_geometry(line_set)
            draw_arraw(segment)
            # 添加端点球体
            for point in segment:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                sphere.translate(point)
                sphere.paint_uniform_color([0, 1, 0])  # 绿色
                vis.add_geometry(sphere)
    
    # 绘制窗口边界
    # 计算窗口的八个角点
    vehicle_dir_2d = current_vehicle_dir[:2]
    normal_dir = np.array([-vehicle_dir_2d[1], vehicle_dir_2d[0]])
    
    # 窗口的四个角点（在车辆坐标系中）
    corners_2d = np.array([
        [-window_length/2, -window_width/2],  # 左下
        [window_length/2, -window_width/2],   # 右下
        [window_length/2, window_width/2],    # 右上
        [-window_length/2, window_width/2]    # 左上
    ])
    
    # 转换到世界坐标系
    world_corners = []
    for corner in corners_2d:
        world_point_2d = window_center[:2] + corner[0] * vehicle_dir_2d + corner[1] * normal_dir
        # 添加Z坐标
        world_point_bottom = np.append(world_point_2d, window_center[2] - window_height/2)
        world_point_top = np.append(world_point_2d, window_center[2] + window_height/2)
        world_corners.extend([world_point_bottom, world_point_top])
    
    world_corners = np.array(world_corners)
    
    # 绘制长方体的12条边
    # 底面四条边
    bottom_lines = [[0, 1], [1, 3], [3, 2], [2, 0]]
    # 顶面四条边
    top_lines = [[4, 5], [5, 7], [7, 6], [6, 4]]
    # 四条竖直边
    vertical_lines = [[0, 4], [1, 5], [2, 6], [3, 7]]
    
    all_lines = bottom_lines + top_lines + vertical_lines
    
    # 创建窗口边界线
    window_lines = o3d.geometry.LineSet()
    window_lines.points = o3d.utility.Vector3dVector(world_corners)
    window_lines.lines = o3d.utility.Vector2iVector(all_lines)
    window_lines.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(all_lines))  # 灰色
    vis.add_geometry(window_lines)
    
    # 绘制车辆方向箭头
    arrow_length = 5.0
    arrow_end_2d = window_center[:2] + vehicle_dir_2d * arrow_length
    arrow_end_3d = np.array([arrow_end_2d[0], arrow_end_2d[1], window_center[2]])
    
    # 创建箭头线段
    arrow_line = o3d.geometry.LineSet()
    arrow_line.points = o3d.utility.Vector3dVector([window_center, arrow_end_3d])
    arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    arrow_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
    vis.add_geometry(arrow_line)
    
    # 创建箭头头部
    arrow_head_length = 1.5
    arrow_head_width = 1.0
    
    arrow_dir = vehicle_dir_2d
    perp_dir = np.array([-arrow_dir[1], arrow_dir[0]])
    
    arrow_tip = arrow_end_3d
    arrow_base_2d = arrow_end_2d - arrow_dir * arrow_head_length
    arrow_base_3d = np.array([arrow_base_2d[0], arrow_base_2d[1], arrow_tip[2]])
    
    arrow_left_2d = arrow_base_2d + perp_dir * arrow_head_width
    arrow_left_3d = np.array([arrow_left_2d[0], arrow_left_2d[1], arrow_tip[2]])
    
    arrow_right_2d = arrow_base_2d - perp_dir * arrow_head_width
    arrow_right_3d = np.array([arrow_right_2d[0], arrow_right_2d[1], arrow_tip[2]])
    
    arrow_head = o3d.geometry.TriangleMesh()
    arrow_head.vertices = o3d.utility.Vector3dVector([arrow_tip, arrow_left_3d, arrow_right_3d])
    arrow_head.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    arrow_head.paint_uniform_color([1, 0, 0])  # 红色
    vis.add_geometry(arrow_head)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 3.0
    opt.line_width = 5.0
    
    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # 打印窗口信息
    print(f"\n=== 窗口 {window_id} 拟合结果 ===")
    print(f"窗口中心: {window_center}")
    print(f"车辆方向: {current_vehicle_dir}")
    print(f"点云数量: {len(window_points)}")
    print(f"虚线线段数: {len(window_segments.get('dashed', []))}")
    print(f"左实线线段数: {len(window_segments.get('left_solid', []))}")
    print(f"右实线线段数: {len(window_segments.get('right_solid', []))}")
    # print("\n=== 可视化图例 ===")
    # print("红色点云: 原始虚线点")
    # print("青色点云: 原始实线点")
    # print("灰色点云: 其他类别点")
    # print("红色粗线+球体: 拟合的虚线线段")
    # print("橙色箭头: 虚线方向")
    # print("蓝色粗线+球体: 拟合的左侧实线线段")
    # print("绿色粗线+球体: 拟合的右侧实线线段")
    # print("灰色线框: 滑动窗口边界")
    # print("红色箭头: 车辆前进方向")
    # print("按 'Q' 退出可视化")
    
    vis.run()
    vis.destroy_window()

def separate_dashed_lanes_by_fitting(dashed_points, vehicle_dir, min_lane_distance=2.5, max_lane_distance=4.5, 
                                     min_points_per_cluster=80):
    """
    专门用于虚线车道线检测和拟合的函数
    
    将点云都投影到车辆的法方向，然后进行DBSCAN聚类，
    取聚类个数大于100的聚类中心，沿着车辆前进方向进行拟合，
    各车道之间的距离需要在2.5到4.5之间
    
    Args:
        dashed_points: 虚线点云 (N, 4)
        vehicle_dir: 车辆方向向量 (2,)
        min_lane_distance: 最小车道间距（米）
        max_lane_distance: 最大车道间距（米）
        min_points_per_cluster: 每个聚类的最小点数
    
    Returns:
        lane_lines: 分离出的车道线列表
    """
    if len(dashed_points) < min_points_per_cluster:
        return [], []
    
    coords_2d = dashed_points[:, :2]
    
    # 1. 计算法向量（垂直于车辆方向，指向车辆左侧）
    normal_vec = np.array([-vehicle_dir[1], vehicle_dir[0]])
    
    # 2. 将点云投影到车辆的法方向（横向投影）
    lateral_proj = np.dot(coords_2d, normal_vec)
    
    print(f"虚线点云总数: {len(dashed_points)}")
    print(f"横向投影范围: [{np.min(lateral_proj):.2f}, {np.max(lateral_proj):.2f}]")
    
    # 3. 使用DBSCAN进行聚类
    from sklearn.cluster import DBSCAN
    
    # 自适应eps参数
    eps_adaptive = np.percentile(np.abs(np.diff(np.sort(lateral_proj))), 75) * 2
    eps_adaptive = max(0.5, min(2.0, eps_adaptive))  # 限制在合理范围内
    
    print(f"DBSCAN参数: eps={eps_adaptive:.3f}, min_samples={min_points_per_cluster}")
    
    # 进行DBSCAN聚类
    db = DBSCAN(eps=eps_adaptive, min_samples=min_points_per_cluster).fit(lateral_proj.reshape(-1, 1))
    labels = db.labels_
    
    # 4. 统计每个聚类的信息
    unique_labels = np.unique(labels[labels != -1])
    print(f"检测到 {len(unique_labels)} 个聚类")
    
    if len(unique_labels) < 1:
        return [], []
    
    # 5. 收集每个聚类的信息
    cluster_info = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = dashed_points[mask]
        cluster_lateral_proj = lateral_proj[mask]
        
        if len(cluster_points) >= min_points_per_cluster:
            # 计算聚类中心（横向位置）
            cluster_center_lateral = np.median(cluster_lateral_proj)
            cluster_info.append({
                'label': label,
                'points': cluster_points,
                'lateral_proj': cluster_lateral_proj,
                'center_lateral': cluster_center_lateral,
                'point_count': len(cluster_points)
            })
            print(f"聚类 {label}: {len(cluster_points)} 点, 横向位置 {cluster_center_lateral:.2f}m")
    
    # 6. 按横向位置排序
    cluster_info.sort(key=lambda x: x['center_lateral'], reverse=True)  # 从车辆左侧到右侧
    
    # 7. 对每个聚类进行拟合
    detected_lanes = []
    for i, cluster in enumerate(cluster_info):
        try:
            # 沿着车辆前进方向进行拟合
            line_result = fit_line_3d_ransac(cluster['points'][:, :3])
            if line_result[0] is not None:
                # 确保拟合的直线段方向与车辆方向一致
                fitted_line = line_result[0]

                # 检查并修正方向
                corrected_line, direction_consistent, angle_diff = check_and_correct_dashed_line_direction(
                    fitted_line, vehicle_dir, angle_threshold=45.0)
                if direction_consistent:
                    detected_lanes.append({
                    'line': corrected_line,
                    'lateral_pos': cluster['center_lateral'],
                    # 'quality': line_quality,
                    'point_count': cluster['point_count']
                    })
                    # lane_lines.append(corrected_line)
                else:
                    # 如果方向不一致，使用备用方法
                    line_points, _ = fit_line_3d_ransac(cluster['points'][:, :3])
                    if line_points is not None:
                        # 对备用方法的结果也进行方向修正
                        corrected_line, _, _ = check_and_correct_dashed_line_direction(
                            line_points, vehicle_dir, angle_threshold=45.0)
                        detected_lanes.append({
                        'line': corrected_line,
                        'lateral_pos': cluster['center_lateral'],
                        # 'quality': line_quality,
                        'point_count': cluster['point_count']
                        })

            # line_result = fit_line_3d_ransac_with_direction_constraint(cluster['points'][:, :3], vehicle_dir)
            # if line_result[0] is not None:
                
            #     fitted_line = line_result[0]
                
            #     # 检查拟合质量
            #     line_quality = calculate_line_quality(fitted_line)
                
            #     detected_lanes.append({
            #         'line': fitted_line,
            #         'lateral_pos': cluster['center_lateral'],
            #         'quality': line_quality,
            #         'point_count': cluster['point_count']
            #     })
                
                print(f"拟合车道线 {i}: 横向位置 {cluster['center_lateral']:.2f}m, 点数 {cluster['point_count']}")
                
        except Exception as e:
            print(f"聚类 {i} 拟合失败: {e}")
            continue
    
    # 8. 验证车道间距 - 集成异常值检测
    valid_indices = set()
    lateral_positions = [lane_info['lateral_pos'] for lane_info in detected_lanes]
    point_counts = [lane_info['point_count'] for lane_info in detected_lanes]
    n = len(detected_lanes)

    if n == 1:
        # 只有一条车道线，直接保留
        valid_indices.add(0)
    else:
        # 计算平均点数，用于判断噪声聚类
        avg_point_count = np.mean(point_counts)
        noise_threshold = avg_point_count * 0.3  # 点数少于平均值的30%认为是噪声
        
        print(f"平均点数: {avg_point_count:.1f}, 噪声阈值: {noise_threshold:.1f}")
        
        # 第一阶段：预标记噪声聚类
        noise_flags = [False] * n
        for i in range(1, n - 1):
            left_distance = abs(lateral_positions[i] - lateral_positions[i-1])
            right_distance = abs(lateral_positions[i] - lateral_positions[i+1])
            if ((left_distance < min_lane_distance or left_distance > max_lane_distance) and
                (right_distance < min_lane_distance or right_distance > max_lane_distance) and
                (point_counts[i] < noise_threshold)):
                noise_flags[i] = True
                print(f"预标记噪声聚类 {i}: distL={left_distance:.2f}, distR={right_distance:.2f}, count={point_counts[i]} < {noise_threshold:.1f}")

        # 第二阶段：在剔除噪声后，使用最近的非噪声邻居做间距校验
        for i in range(n):
            if noise_flags[i]:
                print(f"排除噪声聚类 {i}")
                continue
            
            # 向左寻找最近的非噪声邻居
            left_idx = i - 1
            while left_idx >= 0 and noise_flags[left_idx]:
                left_idx -= 1
            left_valid = False
            if left_idx >= 0:
                left_distance = abs(lateral_positions[i] - lateral_positions[left_idx])
                if min_lane_distance <= left_distance <= max_lane_distance:
                    left_valid = True
                else:
                    print(f"车道间距不合理: {left_distance:.2f}m (车道 {left_idx} 和 {i})")
            
            # 向右寻找最近的非噪声邻居
            right_idx = i + 1
            while right_idx < n and noise_flags[right_idx]:
                right_idx += 1
            right_valid = False
            if right_idx < n:
                right_distance = abs(lateral_positions[i] - lateral_positions[right_idx])
                if min_lane_distance <= right_distance <= max_lane_distance:
                    right_valid = True
                else:
                    print(f"车道间距不合理: {right_distance:.2f}m (车道 {i} 和 {right_idx})")
            
            if left_valid or right_valid:
                valid_indices.add(i)
                print(f"保留车道 {i}: 与最近的非噪声邻居保持合理间距")
            else:
                print(f"排除车道 {i}: 与最近的非噪声邻居间距都不合理")

    valid_lanes = []
    cluster_points = []
    for idx in sorted(valid_indices):
        valid_lanes.append(detected_lanes[idx]['line'])
        cluster_points.append(detected_lanes[idx]['point_count'])

    print(f"最终检测到 {len(valid_lanes)} 条车道线")
    return valid_lanes, cluster_points

def clip_line_to_window(line, window_start, window_end, vehicle_dir):
    """
    将一条直线段裁剪到窗口长度的0.9倍范围内。
    line: (2,3) ndarray，直线段的起点和终点
    window_start, window_end: 沿vehicle_dir的投影起止值
    vehicle_dir: (2,) 车辆方向向量，已归一化
    返回: 裁剪后的(2,3)线段
    """
    import numpy as np
    p0, p1 = line[0][:2], line[1][:2]
    z0, z1 = line[0][2], line[1][2]
    # 计算起点和终点在车辆方向上的投影
    proj0 = np.dot(p0, vehicle_dir)
    proj1 = np.dot(p1, vehicle_dir)
    
    # 计算窗口中心点
    window_center = (window_start + window_end) / 2
    window_length = window_end - window_start
    # 裁剪到窗口长度的0.9倍
    clip_length = window_length * 0.9
    clip_start = window_center - clip_length / 2
    clip_end = window_center + clip_length / 2
    
    # 计算裁剪后的投影
    clipped_proj0 = np.clip(proj0, clip_start, clip_end)
    clipped_proj1 = np.clip(proj1, clip_start, clip_end)
    
    # 计算裁剪后点的实际坐标
    def get_point(orig, proj, clipped_proj, z):
        delta = clipped_proj - proj
        return np.array([orig[0] + delta * vehicle_dir[0], orig[1] + delta * vehicle_dir[1], z])
    new_p0 = get_point(p0, proj0, clipped_proj0, z0)
    new_p1 = get_point(p1, proj1, clipped_proj1, z1)
    return np.vstack([new_p0, new_p1])

if __name__ == "__main__":
    main()
