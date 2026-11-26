#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
道路结构分析器
用于分析道路结构，识别直线段和转弯段，并分割车道线
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os
import open3d as o3d
import sys

def quaternion_to_yaw(qx, qy, qz, qw):
    """
    四元数转yaw角（弧度）
    
    Args:
        qx, qy, qz, qw: 四元数分量
    
    Returns:
        yaw: yaw角（弧度）
    """
    # 参考 https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入相关函数
try:
    from process_lane_segments import (
        fit_line_ransac, fit_lane_segments_adaptive,
        filter_points_by_class, line_to_line_lateral_distance
    )
except ImportError:
    print("Warning: 无法导入 process_lane_segments 模块，某些功能可能不可用")

# 定义类别ID
LANE_SOLID_CLASS = 1
LANE_DASHED_CLASS = 2
LANE_ARROW_CLASS = 3
STOP_LINE_CLASS = 5

def analyze_road_structure_from_trajectory(poses, window_size=10, angle_threshold=15.0, curvature_threshold=0.005, ignore_start_end_ratio=0.1):
    """
    通过车辆轨迹分析道路结构，重点识别转弯段，忽略开始和结束部分
    
    Args:
        poses: (N, 6) 车辆位姿 [x, y, z, roll, pitch, yaw]
        window_size: 滑动窗口大小
        angle_threshold: 转弯角度阈值（度）
        curvature_threshold: 曲率阈值
        ignore_start_end_ratio: 忽略开始和结束部分的比例
    
    Returns:
        road_segments: list of dict, 道路段信息
    """
    if len(poses) < window_size:
        return []
    
    # 1. 计算每个点的航向角变化和曲率
    heading_changes = []
    curvatures = []
    
    for i in range(1, len(poses)):
        # 计算航向角变化
        prev_yaw = poses[i-1][5]
        curr_yaw = poses[i][5]
        
        # 处理角度跳变
        angle_diff = curr_yaw - prev_yaw
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        heading_changes.append(abs(angle_diff))
    
    # 计算每个点的曲率
    for i in range(1, len(poses) - 1):
        p1 = poses[i-1][:2]
        p2 = poses[i][:2]
        p3 = poses[i+1][:2]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross_product = np.cross(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            curvature = np.abs(cross_product) / (v1_norm * v2_norm * np.linalg.norm(v1 + v2))
            curvatures.append(curvature)
        else:
            curvatures.append(0.0)
    
    # 补充首尾点的曲率
    curvatures = [0.0] + curvatures + [0.0]
    
    # 2. 识别转弯段
    curve_segments = []
    in_curve = False
    curve_start = 0
    
    for i in range(window_size, len(poses) - window_size):
        # 计算窗口内的平均航向角变化和曲率
        window_changes = heading_changes[i-window_size:i]
        window_curvatures = curvatures[i-window_size:i+window_size]
        
        avg_heading_change = np.mean(window_changes)
        avg_curvature = np.mean(window_curvatures)
        max_curvature = np.max(window_curvatures)
        
        # 判断是否为转弯段
        is_curve = (avg_heading_change > np.radians(angle_threshold) or 
                   avg_curvature > curvature_threshold or 
                   max_curvature > curvature_threshold * 2)
        
        if is_curve and not in_curve:
            # 开始转弯段
            in_curve = True
            curve_start = i - window_size
        elif not is_curve and in_curve:
            # 结束转弯段
            in_curve = False
            curve_end = i + window_size
            
            # 确保转弯段有足够长度
            if curve_end - curve_start >= 5:  # 至少5个点
                curve_segments.append({
                    'start_idx': curve_start,
                    'end_idx': curve_end,
                    'start_pose': poses[curve_start],
                    'end_pose': poses[curve_end],
                    'type': 'curve',
                    'avg_heading_change': np.mean(heading_changes[curve_start:curve_end]),
                    'avg_curvature': np.mean(curvatures[curve_start:curve_end]),
                    'max_curvature': np.max(curvatures[curve_start:curve_end])
                })
    
    # 处理最后一个转弯段
    if in_curve:
        curve_end = len(poses) - 1
        if curve_end - curve_start >= 5:
            curve_segments.append({
                'start_idx': curve_start,
                'end_idx': curve_end,
                'start_pose': poses[curve_start],
                'end_pose': poses[curve_end],
                'type': 'curve',
                'avg_heading_change': np.mean(heading_changes[curve_start:curve_end]),
                'avg_curvature': np.mean(curvatures[curve_start:curve_end]),
                'max_curvature': np.max(curvatures[curve_start:curve_end])
            })
    
    # 3. 构建道路段（直线段 + 转弯段）
    road_segments = []
    
    # 添加起始直线段
    if curve_segments and curve_segments[0]['start_idx'] > 0:
        road_segments.append({
            'type': 'straight',
            'start_idx': 0,
            'end_idx': curve_segments[0]['start_idx'],
            'start_pose': poses[0],
            'end_pose': poses[curve_segments[0]['start_idx']],
            'avg_heading': np.mean(heading_changes[0:curve_segments[0]['start_idx']]),
            'curvature': np.mean(curvatures[0:curve_segments[0]['start_idx']])
        })
    
    # 交替添加转弯段和直线段
    for i, curve_seg in enumerate(curve_segments):
        # 添加转弯段
        road_segments.append(curve_seg)
        
        # 添加转弯段后的直线段
        if i < len(curve_segments) - 1:
            next_curve = curve_segments[i + 1]
            if next_curve['start_idx'] > curve_seg['end_idx']:
                road_segments.append({
                    'type': 'straight',
                    'start_idx': curve_seg['end_idx'],
                    'end_idx': next_curve['start_idx'],
                    'start_pose': poses[curve_seg['end_idx']],
                    'end_pose': poses[next_curve['start_idx']],
                    'avg_heading': np.mean(heading_changes[curve_seg['end_idx']:next_curve['start_idx']]),
                    'curvature': np.mean(curvatures[curve_seg['end_idx']:next_curve['start_idx']])
                })
        else:
            # 最后一个转弯段后的直线段
            if curve_seg['end_idx'] < len(poses) - 1:
                road_segments.append({
                    'type': 'straight',
                    'start_idx': curve_seg['end_idx'],
                    'end_idx': len(poses) - 1,
                    'start_pose': poses[curve_seg['end_idx']],
                    'end_pose': poses[-1],
                    'avg_heading': np.mean(heading_changes[curve_seg['end_idx']:]),
                    'curvature': np.mean(curvatures[curve_seg['end_idx']:])
                })
    
    # 如果没有转弯段，整个轨迹都是直线
    if not curve_segments:
        road_segments.append({
            'type': 'straight',
            'start_idx': 0,
            'end_idx': len(poses) - 1,
            'start_pose': poses[0],
            'end_pose': poses[-1],
            'avg_heading': np.mean(heading_changes),
            'curvature': np.mean(curvatures)
        })
    
    return road_segments

def calculate_trajectory_curvature(poses):
    """计算轨迹的曲率"""
    if len(poses) < 3:
        return 0.0
    
    curvatures = []
    for i in range(1, len(poses) - 1):
        p1 = poses[i-1][:2]  # 前一个点
        p2 = poses[i][:2]    # 当前点
        p3 = poses[i+1][:2]  # 后一个点
        
        # 计算三点构成的圆的曲率
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算叉积和模长
        cross_product = np.cross(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            # 曲率 = |叉积| / (|v1| * |v2| * |v1 + v2|)
            curvature = np.abs(cross_product) / (v1_norm * v2_norm * np.linalg.norm(v1 + v2))
            curvatures.append(curvature)
    
    return np.mean(curvatures) if curvatures else 0.0

def is_dashed_line_straight_or_curved(points, threshold_ratio=0.1):
    """
    判断虚线点云构成的形状是直线还是曲线
    
    Args:
        points: (N, 3) 虚线点云
        threshold_ratio: 判断阈值，直线拟合误差与曲线拟合误差的比值
    
    Returns:
        is_straight: bool, True表示直线，False表示曲线
        straight_error: 直线拟合误差
        curved_error: 曲线拟合误差
    """
    if len(points) < 10:
        return True, 0, 0
    
    # 1. 直线拟合
    try:
        line, straight_error = fit_line_ransac(points)
        if line is None:
            return True, float('inf'), float('inf')
    except:
        return True, float('inf'), float('inf')
    
    # 2. 多项式曲线拟合（二次曲线）
    try:
        # 将点投影到主方向上进行曲线拟合
        coords = points[:, :2]
        
        # 计算主方向
        if len(coords) >= 2:
            # 使用PCA找到主方向
            pca = PCA(n_components=2)
            pca.fit(coords)
            principal_direction = pca.components_[0]
            
            # 将点投影到主方向上
            projected_coords = np.dot(coords, principal_direction)
            
            # 对投影坐标进行排序
            sorted_indices = np.argsort(projected_coords)
            sorted_projected = projected_coords[sorted_indices]
            sorted_points = coords[sorted_indices]
            
            # 二次多项式拟合
            x = sorted_projected
            y = sorted_points[:, 1]  # 使用y坐标作为因变量
            
            # 拟合二次曲线: y = ax² + bx + c
            coeffs = np.polyfit(x, y, 2)
            y_pred = np.polyval(coeffs, x)
            
            # 计算曲线拟合误差
            curved_error = np.mean((y - y_pred) ** 2)
            
        else:
            curved_error = float('inf')
            
    except:
        curved_error = float('inf')
    
    # 3. 比较拟合误差
    # 确保误差是标量
    if isinstance(straight_error, np.ndarray):
        straight_error = np.mean(straight_error) if len(straight_error) > 0 else float('inf')
    if isinstance(curved_error, np.ndarray):
        curved_error = np.mean(curved_error) if len(curved_error) > 0 else float('inf')
    
    if straight_error == 0 and curved_error == 0:
        return True, 0, 0
    
    if curved_error == 0:
        return False, straight_error, curved_error
    
    error_ratio = straight_error / curved_error
    
    # 如果直线拟合误差比曲线拟合误差小很多，认为是直线
    is_straight = error_ratio < threshold_ratio
    
    return is_straight, straight_error, curved_error

def analyze_dashed_line_shape(points, visualize=False):
    """
    分析虚线点云的形状特征
    
    Args:
        points: (N, 3) 虚线点云
        visualize: 是否可视化分析结果
    
    Returns:
        shape_info: dict, 包含形状分析结果
    """
    if len(points) < 10:
        return {"type": "unknown", "confidence": 0.0, "error_ratio": 1.0}
    
    # 1. 判断直线还是曲线
    is_straight, straight_error, curved_error = is_dashed_line_straight_or_curved(points)
    
    # 2. 计算曲率特征
    try:
        coords = points[:, :2]
        
        # 使用PCA找到主方向
        pca = PCA(n_components=2)
        pca.fit(coords)
        principal_direction = pca.components_[0]
        
        # 将点投影到主方向上
        projected_coords = np.dot(coords, principal_direction)
        
        # 对投影坐标进行排序
        sorted_indices = np.argsort(projected_coords)
        sorted_projected = projected_coords[sorted_indices]
        sorted_points = coords[sorted_indices]
        
        # 计算离散曲率
        if len(sorted_points) >= 3:
            # 使用三点法计算曲率
            curvatures = []
            for i in range(1, len(sorted_points) - 1):
                p1 = sorted_points[i-1]
                p2 = sorted_points[i]
                p3 = sorted_points[i+1]
                
                # 计算三点构成的圆的曲率
                v1 = p2 - p1
                v2 = p3 - p2
                
                # 计算叉积和模长
                cross_product = np.cross(v1, v2)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    # 曲率 = |叉积| / (|v1| * |v2| * |v1 + v2|)
                    curvature = np.abs(cross_product) / (v1_norm * v2_norm * np.linalg.norm(v1 + v2))
                    curvatures.append(curvature)
            
            avg_curvature = np.mean(curvatures) if curvatures else 0
            max_curvature = np.max(curvatures) if curvatures else 0
        else:
            avg_curvature = 0
            max_curvature = 0
            
    except:
        avg_curvature = 0
        max_curvature = 0
    
    # 3. 计算方向一致性
    try:
        line, _ = fit_line_ransac(points)
        if line is not None:
            line_direction = line[1][:2] - line[0][:2]
            line_direction = line_direction / np.linalg.norm(line_direction)
            
            # 计算每个点到直线的方向一致性
            direction_consistencies = []
            for point in points[:, :2]:
                # 计算点到直线的投影
                line_point = line[0][:2]
                point_vector = point - line_point
                projection = np.dot(point_vector, line_direction) * line_direction
                
                # 计算实际方向与投影方向的夹角余弦
                if np.linalg.norm(point_vector) > 0:
                    cos_angle = np.dot(point_vector, projection) / (np.linalg.norm(point_vector) * np.linalg.norm(projection))
                    direction_consistencies.append(abs(cos_angle))
            
            avg_direction_consistency = np.mean(direction_consistencies) if direction_consistencies else 0
        else:
            avg_direction_consistency = 0
    except:
        avg_direction_consistency = 0
    
    # 4. 综合判断
    error_ratio = straight_error / curved_error if curved_error > 0 else float('inf')
    
    # 判断形状类型
    if is_straight:
        shape_type = "straight"
        confidence = min(1.0, 1.0 / error_ratio) if error_ratio > 0 else 1.0
    else:
        shape_type = "curved"
        confidence = min(1.0, error_ratio) if error_ratio < float('inf') else 0.5
    
    # 根据曲率进一步细化
    if shape_type == "curved":
        if avg_curvature > 0.01:  # 高曲率
            shape_type = "sharp_curve"
        elif avg_curvature > 0.001:  # 中等曲率
            shape_type = "gentle_curve"
        else:  # 低曲率
            shape_type = "slight_curve"
    
    shape_info = {
        "type": shape_type,
        "confidence": confidence,
        "error_ratio": error_ratio,
        "straight_error": straight_error,
        "curved_error": curved_error,
        "avg_curvature": avg_curvature,
        "max_curvature": max_curvature,
        "direction_consistency": avg_direction_consistency,
        "point_count": len(points)
    }
    
    # 可视化分析结果
    if visualize:
        print(f"虚线形状分析结果:")
        print(f"  类型: {shape_type}")
        print(f"  置信度: {confidence:.3f}")
        print(f"  误差比: {error_ratio:.3f}")
        print(f"  平均曲率: {avg_curvature:.6f}")
        print(f"  最大曲率: {max_curvature:.6f}")
        print(f"  方向一致性: {avg_direction_consistency:.3f}")
        print(f"  点数: {len(points)}")
    
    return shape_info

def project_points_to_trajectory(points, poses, road_segments):
    """
    将点云投影到轨迹上，根据道路段分类
    
    Args:
        points: (N, 3) 点云
        poses: (M, 6) 车辆位姿
        road_segments: 道路段信息
    
    Returns:
        projected_segments: list of dict, 投影后的段信息
    """
    projected_segments = []
    
    for segment in road_segments:
        segment_points = []
        start_pose = segment['start_pose']
        end_pose = segment['end_pose']
        
        # 计算段的方向向量
        segment_direction = end_pose[:2] - start_pose[:2]
        segment_length = np.linalg.norm(segment_direction)
        if segment_length > 0:
            segment_direction = segment_direction / segment_length
        
        # 计算段的法向量
        segment_normal = np.array([-segment_direction[1], segment_direction[0]])
        
        # 将点投影到当前段
        for point in points:
            # 计算点到段起点的向量
            point_vector = point[:2] - start_pose[:2]
            
            # 计算点在段方向上的投影
            projection_along = np.dot(point_vector, segment_direction)
            
            # 计算点到段的垂直距离
            projection_perpendicular = np.dot(point_vector, segment_normal)
            
            # 判断点是否属于当前段
            if 0 <= projection_along <= segment_length and abs(projection_perpendicular) < 20.0:  # 20米宽度
                segment_points.append(point)
        
        projected_segments.append({
            'road_type': segment['type'],
            'points': np.array(segment_points) if segment_points else np.empty((0, 3)),
            'segment_info': segment
        })
    
    return projected_segments

def merge_adjacent_straight_segments(straight_segments, min_length=50.0):
    """
    合并相邻的直线段
    
    Args:
        straight_segments: list of arrays, 直线段点云
        min_length: 最小合并长度（米）
    
    Returns:
        merged_segments: list of arrays, 合并后的直线段
    """
    if len(straight_segments) <= 1:
        return straight_segments
    
    merged_segments = []
    current_segment = straight_segments[0].copy()
    
    for i in range(1, len(straight_segments)):
        next_segment = straight_segments[i]
        
        # 计算当前段和下一段的拟合直线
        if len(current_segment) >= 2 and len(next_segment) >= 2:
            current_line, _ = fit_line_ransac(current_segment)
            next_line, _ = fit_line_ransac(next_segment)
            
            if current_line is not None and next_line is not None:
                # 计算两条线的方向一致性
                current_direction = current_line[1][:2] - current_line[0][:2]
                next_direction = next_line[1][:2] - next_line[0][:2]
                
                current_direction = current_direction / np.linalg.norm(current_direction)
                next_direction = next_direction / np.linalg.norm(next_direction)
                
                direction_consistency = abs(np.dot(current_direction, next_direction))
                
                # 计算两条线之间的横向距离
                lateral_distance = line_to_line_lateral_distance(current_line, next_line)
                
                # 判断是否可以合并
                if direction_consistency > 0.9 and lateral_distance < 2.0:
                    # 合并段
                    current_segment = np.vstack([current_segment, next_segment])
                else:
                    # 结束当前段，开始新段
                    if len(current_segment) > 0:
                        merged_segments.append(current_segment)
                    current_segment = next_segment.copy()
            else:
                # 无法拟合直线，直接合并
                current_segment = np.vstack([current_segment, next_segment])
        else:
            # 点数不足，直接合并
            current_segment = np.vstack([current_segment, next_segment])
    
    # 添加最后一段
    if len(current_segment) > 0:
        merged_segments.append(current_segment)
    
    # 过滤掉太短的段
    filtered_segments = []
    for segment in merged_segments:
        if len(segment) >= 2:
            # 计算段的长度
            line, _ = fit_line_ransac(segment)
            if line is not None:
                segment_length = np.linalg.norm(line[1][:2] - line[0][:2])
                if segment_length >= min_length:
                    filtered_segments.append(segment)
    
    return filtered_segments

def segment_dashed_lane_by_road_structure(dashed_points, poses, min_straight_length=50.0, 
                                        angle_threshold=15.0, max_gap=20, visualize=False):
    """
    根据道路结构分割虚线车道线点云，支持转弯合并
    
    Args:
        dashed_points: (N, 3) 虚线点云
        poses: (M, 6) 车辆位姿
        min_straight_length: 最小直线段长度（米）
        angle_threshold: 转弯角度阈值（度）
        max_gap: 合并转弯的最大间隔
        visualize: 是否可视化
    
    Returns:
        straight_segments: list of arrays, 直线段点云
        curve_segments: list of arrays, 曲线段点云
        road_structure: dict, 道路结构信息
    """
    if len(dashed_points) == 0 or len(poses) == 0:
        return [], [], {}
    
    # 1. 检测并合并转弯
    merged_turns, straight_segments, original_turns = detect_and_merge_turns(
        poses, angle_threshold=angle_threshold, max_gap=max_gap
    )
    
    # 2. 构建道路段信息
    road_segments = []
    
    # 添加直线段
    for i, seg in enumerate(straight_segments):
        road_segments.append({
            'type': 'straight',
            'start_idx': seg['start_idx'],
            'end_idx': seg['end_idx'],
            'start_pose': seg['start_pose'],
            'end_pose': seg['end_pose']
        })
    
    # 添加转弯段
    for i, turn in enumerate(merged_turns):
        road_segments.append({
            'type': 'curve',
            'start_idx': turn['start_idx'],
            'end_idx': turn['end_idx'],
            'start_pose': turn['start_pose'],
            'end_pose': turn['end_pose'],
            'turn_angle': turn['turn_angle'],
            'avg_curvature': turn['avg_curvature']
        })
    
    # 按起始索引排序
    road_segments.sort(key=lambda x: x['start_idx'])
    
    # 3. 将虚线点云投影到轨迹上
    projected_segments = project_points_to_trajectory(dashed_points, poses, road_segments)
    
    # 4. 根据道路结构分割虚线点云
    straight_segments = []
    curve_segments = []
    
    for segment_info in projected_segments:
        if segment_info['road_type'] == 'straight':
            if len(segment_info['points']) > 0:
                straight_segments.append(segment_info['points'])
        else:
            if len(segment_info['points']) > 0:
                curve_segments.append(segment_info['points'])
    
    # 5. 合并相邻的直线段
    merged_straight_segments = merge_adjacent_straight_segments(straight_segments, min_straight_length)
    
    # 6. 构建道路结构信息
    road_structure = {
        'total_segments': len(road_segments),
        'straight_segments': len(merged_straight_segments),
        'curve_segments': len(curve_segments),
        'road_segments': road_segments,
        'original_turns': original_turns,
        'merged_turns': merged_turns,
        'total_dashed_points': len(dashed_points),
        'straight_points': sum(len(seg) for seg in merged_straight_segments),
        'curve_points': sum(len(seg) for seg in curve_segments)
    }
    
    if visualize:
        print(f"道路结构分析结果:")
        print(f"  原始转弯数: {len(original_turns)}")
        print(f"  合并后转弯数: {len(merged_turns)}")
        print(f"  总道路段数: {road_structure['total_segments']}")
        print(f"  直线段数: {road_structure['straight_segments']}")
        print(f"  曲线段数: {road_structure['curve_segments']}")
        print(f"  虚线总点数: {road_structure['total_dashed_points']}")
        print(f"  直线段点数: {road_structure['straight_points']}")
        print(f"  曲线段点数: {road_structure['curve_points']}")
        
        # 可视化道路结构（包含合并对比）
        visualize_road_structure(road_segments, poses, original_turns, merged_turns)
    
    return merged_straight_segments, curve_segments, road_structure

def calculate_polyline_length(polyline):
    """计算折线长度"""
    if len(polyline) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(polyline) - 1):
        segment_length = np.linalg.norm(polyline[i+1][:2] - polyline[i][:2])
        total_length += segment_length
    
    return total_length

def merge_segments_to_polyline(segments):
    """合并多个线段为一个折线"""
    if len(segments) == 0:
        return np.array([])
    
    # 3. 构建polyline
    polyline = []
    
    for i, info in enumerate(segments):

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
    
    
    
    if len(polyline) < 2:
        return None
    
    return np.array(polyline)


def analyze_and_segment_road_lanes(points, poses, dashed_class=2, solid_class=1, 
                                 min_straight_length=50.0, angle_threshold=15.0, 
                                 visualize=False):
    """
    综合分析道路结构并分割车道线
    
    Args:
        points: (N, 3) 完整点云
        poses: (M, 6) 车辆位姿
        dashed_class: 虚线类别ID
        solid_class: 实线类别ID
        min_straight_length: 最小直线段长度
        angle_threshold: 转弯角度阈值
        visualize: 是否可视化
    
    Returns:
        road_analysis: dict, 道路分析结果
    """
    # 1. 提取虚线点云
    dashed_points = filter_points_by_class(points, [dashed_class])
    print(f"虚线点数: {len(dashed_points)}, 位姿点数: {len(poses)}")
    
    # 2. 根据道路结构分割虚线
    straight_segments, curve_segments, road_structure = segment_dashed_lane_by_road_structure(
        dashed_points, poses, min_straight_length, angle_threshold, max_gap=20, visualize=visualize
    )
    
    # 3. 对每个直线段进行车道线拟合
    straight_lane_results = []
    for i, segment in enumerate(straight_segments):
        if len(segment) > 0:
            # 分析直线段的形状
            shape_info = analyze_dashed_line_shape(segment, visualize=False)
            
            # 拟合车道线
            if shape_info['type'] == 'straight':
                # 直线拟合
                line, _ = fit_line_ransac(segment)
                if line is not None:
                    straight_lane_results.append({
                        'segment_id': i,
                        'type': 'straight',
                        'line': line,
                        'points': segment,
                        'shape_info': shape_info,
                        'length': np.linalg.norm(line[1][:2] - line[0][:2])
                    })
            else:
                # 曲线拟合
                segments = fit_lane_segments_adaptive(segment, np.array([1, 0]), block_size=15.0, min_points=50)
                if len(segments) > 0:
                    polyline = merge_segments_to_polyline(segments)
                    straight_lane_results.append({
                        'segment_id': i,
                        'type': 'curved',
                        'line': polyline,
                        'points': segment,
                        'shape_info': shape_info,
                        'length': calculate_polyline_length(polyline)
                    })
    
    # 4. 对曲线段进行拟合
    curve_lane_results = []
    for i, segment in enumerate(curve_segments):
        if len(segment) > 0:
            shape_info = analyze_dashed_line_shape(segment, visualize=False)
            segments = fit_lane_segments_adaptive(segment, np.array([1, 0]), block_size=10.0, min_points=30)
            if len(segments) > 0:
                polyline = merge_segments_to_polyline(segments)
                curve_lane_results.append({
                    'segment_id': i,
                    'type': 'curve',
                    'line': polyline,
                    'points': segment,
                    'shape_info': shape_info,
                    'length': calculate_polyline_length(polyline)
                })
    
    # 5. 构建分析结果
    road_analysis = {
        'road_structure': road_structure,
        'straight_lanes': straight_lane_results,
        'curve_lanes': curve_lane_results,
        'total_straight_lanes': len(straight_lane_results),
        'total_curve_lanes': len(curve_lane_results),
        'total_lane_length': sum(lane['length'] for lane in straight_lane_results + curve_lane_results)
    }
    
    if visualize:
        print(f"\n道路车道线分析结果:")
        print(f"  直线段车道线数: {road_analysis['total_straight_lanes']}")
        print(f"  曲线段车道线数: {road_analysis['total_curve_lanes']}")
        print(f"  总车道线长度: {road_analysis['total_lane_length']:.2f}米")
        
        for i, lane in enumerate(straight_lane_results):
            print(f"  直线段{i}: 长度={lane['length']:.2f}m, 形状={lane['shape_info']['type']}")
        
        for i, lane in enumerate(curve_lane_results):
            print(f"  曲线段{i}: 长度={lane['length']:.2f}m, 形状={lane['shape_info']['type']}")
    
    return road_analysis

def visualize_road_structure(road_segments, poses, original_turns=None, merged_turns=None):
    """可视化道路结构，重点显示转弯识别结果，支持显示合并前后的对比"""
    try:
        # 提取轨迹点
        trajectory_x = [pose[0] for pose in poses]
        trajectory_y = [pose[1] for pose in poses]
        
        # 创建子图
        if original_turns is not None and merged_turns is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            ax2 = None
        
        # 绘制完整轨迹
        ax1.plot(trajectory_x, trajectory_y, 'b-', alpha=0.3, linewidth=1, label='Vehicle Trajectory')
        
        # 绘制道路段
        colors = {'straight': 'green', 'curve': 'red', 'arrow': 'yellow'}
        
        for i, segment in enumerate(road_segments):
            start_pose = segment['start_pose']
            end_pose = segment['end_pose']
            segment_type = segment['type']
            
            color = colors.get(segment_type, 'gray')
            linewidth = 4 if segment_type == 'curve' else 2  # 转弯段用粗线
            
            # 绘制段
            ax1.plot([start_pose[0], end_pose[0]], [start_pose[1], end_pose[1]], 
                    color=color, linewidth=linewidth, label=f'{segment_type}_{i}')
            
            # 在转弯段中间添加标记
            if segment_type == 'curve':
                mid_x = (start_pose[0] + end_pose[0]) / 2
                mid_y = (start_pose[1] + end_pose[1]) / 2
                ax1.plot(mid_x, mid_y, 'ro', markersize=8, markeredgecolor='black')
                ax1.text(mid_x, mid_y + 2, f'Turn {i}', fontsize=10, ha='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Road Structure Analysis - Final Result')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 添加统计信息
        straight_count = sum(1 for seg in road_segments if seg['type'] == 'straight')
        curve_count = sum(1 for seg in road_segments if seg['type'] == 'curve')
        ax1.text(0.02, 0.98, f'Straight segments: {straight_count}\nCurve segments: {curve_count}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # 如果有对比数据，绘制对比图
        if ax2 is not None and original_turns is not None and merged_turns is not None:
            ax2.plot(trajectory_x, trajectory_y, 'b-', alpha=0.3, linewidth=1, label='Vehicle Trajectory')
            
            # 绘制原始转弯（小圆圈）
            for i, turn in enumerate(original_turns):
                start_pose = turn['start_pose']
                end_pose = turn['end_pose']
                ax2.plot([start_pose[0], end_pose[0]], [start_pose[1], end_pose[1]], 
                        color='orange', linewidth=3, alpha=0.7, label=f'Original Turn {i}' if i < 3 else "")
                
                # 标记原始转弯
                mid_x = (start_pose[0] + end_pose[0]) / 2
                mid_y = (start_pose[1] + end_pose[1]) / 2
                ax2.plot(mid_x, mid_y, 'yo', markersize=6, markeredgecolor='black')
            
            # 绘制合并后的转弯（大圆圈）
            for i, turn in enumerate(merged_turns):
                start_pose = turn['start_pose']
                end_pose = turn['end_pose']
                ax2.plot([start_pose[0], end_pose[0]], [start_pose[1], end_pose[1]], 
                        color='red', linewidth=5, label=f'Merged Turn {i}' if i < 3 else "")
                
                # 标记合并后的转弯
                mid_x = (start_pose[0] + end_pose[0]) / 2
                mid_y = (start_pose[1] + end_pose[1]) / 2
                ax2.plot(mid_x, mid_y, 'ro', markersize=10, markeredgecolor='black')
                ax2.text(mid_x, mid_y + 3, f'Merged {i}', fontsize=12, ha='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Turn Merging Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            # 添加合并统计信息
            ax2.text(0.02, 0.98, f'Original turns: {len(original_turns)}\nMerged turns: {len(merged_turns)}', 
                    transform=ax2.transAxes, fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for visualization")

def load_poses(pose_file):
    """加载位姿文件 (TUM格式: timestamp x y z qx qy qz qw)"""
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
                yaw = quaternion_to_yaw(qx, qy, qz, qw)
                poses.append([x, y, z, 0.0, 0.0, yaw])
    return np.array(poses)

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
    
def detect_turns_from_trajectory(poses, window_size=10, angle_threshold=15.0, curvature_threshold=0.005, min_turn_length=5, ignore_start_end_ratio=0.1):
    """
    专门检测轨迹中的转弯段，忽略开始和结束部分
    
    Args:
        poses: (N, 6) 车辆位姿
        window_size: 滑动窗口大小
        angle_threshold: 转弯角度阈值（度）
        curvature_threshold: 曲率阈值
        min_turn_length: 最小转弯段长度
        ignore_start_end_ratio: 忽略开始和结束部分的比例
    
    Returns:
        turns: list of dict, 转弯段信息
        straight_segments: list of dict, 直线段信息
    """
    if len(poses) < window_size:
        return [], []
    
    # 计算忽略的开始和结束索引
    ignore_start = int(len(poses) * ignore_start_end_ratio)
    ignore_end = len(poses) - ignore_start
    
    print(f"忽略轨迹开始和结束部分: 0-{ignore_start}, {ignore_end}-{len(poses)}")
    
    # 计算航向角变化和曲率
    heading_changes = []
    curvatures = []
    
    for i in range(1, len(poses)):
        prev_yaw = poses[i-1][5]
        curr_yaw = poses[i][5]
        
        angle_diff = curr_yaw - prev_yaw
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        heading_changes.append(abs(angle_diff))
    
    # 计算曲率
    for i in range(1, len(poses) - 1):
        p1 = poses[i-1][:2]
        p2 = poses[i][:2]
        p3 = poses[i+1][:2]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross_product = np.cross(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            curvature = np.abs(cross_product) / (v1_norm * v2_norm * np.linalg.norm(v1 + v2))
            curvatures.append(curvature)
        else:
            curvatures.append(0.0)
    
    curvatures = [0.0] + curvatures + [0.0]
    
    # 检测转弯段（只在有效范围内检测）
    turns = []
    in_turn = False
    turn_start = 0
    
    # 调整检测范围，避开开始和结束部分
    start_idx = max(window_size, ignore_start)
    end_idx = min(len(poses) - window_size, ignore_end)
    
    for i in range(start_idx, end_idx):
        window_changes = heading_changes[i-window_size:i]
        window_curvatures = curvatures[i-window_size:i+window_size]
        
        avg_heading_change = np.mean(window_changes)
        avg_curvature = np.mean(window_curvatures)
        max_curvature = np.max(window_curvatures)
        
        # 判断是否为转弯
        is_turn = (avg_heading_change > np.radians(angle_threshold) or 
                  avg_curvature > curvature_threshold or 
                  max_curvature > curvature_threshold * 2)
        
        if is_turn and not in_turn:
            in_turn = True
            turn_start = i - window_size
        elif not is_turn and in_turn:
            in_turn = False
            turn_end = i + window_size
            
            # 确保转弯段不在忽略区域内
            if (turn_start >= ignore_start and turn_end <= ignore_end and 
                turn_end - turn_start >= min_turn_length):
                turns.append({
                    'start_idx': turn_start,
                    'end_idx': turn_end,
                    'start_pose': poses[turn_start],
                    'end_pose': poses[turn_end],
                    'avg_heading_change': np.mean(heading_changes[turn_start:turn_end]),
                    'avg_curvature': np.mean(curvatures[turn_start:turn_end]),
                    'max_curvature': np.max(curvatures[turn_start:turn_end]),
                    'turn_angle': np.sum(heading_changes[turn_start:turn_end]) * np.sign(np.mean(heading_changes[turn_start:turn_end]))
                })
    
    # 处理最后一个转弯（如果还在转弯状态）
    if in_turn:
        turn_end = min(len(poses) - 1, ignore_end)
        if (turn_start >= ignore_start and turn_end <= ignore_end and 
            turn_end - turn_start >= min_turn_length):
            turns.append({
                'start_idx': turn_start,
                'end_idx': turn_end,
                'start_pose': poses[turn_start],
                'end_pose': poses[turn_end],
                'avg_heading_change': np.mean(heading_changes[turn_start:turn_end]),
                'avg_curvature': np.mean(curvatures[turn_start:turn_end]),
                'max_curvature': np.max(curvatures[turn_start:turn_end]),
                'turn_angle': np.sum(heading_changes[turn_start:turn_end]) * np.sign(np.mean(heading_changes[turn_start:turn_end]))
            })
    
    print(f"在有效范围内检测到 {len(turns)} 个转弯")
    
    # 构建直线段
    straight_segments = []
    
    if not turns:
        # 没有转弯，整个轨迹都是直线
        straight_segments.append({
            'start_idx': 0,
            'end_idx': len(poses) - 1,
            'start_pose': poses[0],
            'end_pose': poses[-1]
        })
    else:
        # 添加起始直线段
        if turns[0]['start_idx'] > 0:
            straight_segments.append({
                'start_idx': 0,
                'end_idx': turns[0]['start_idx'],
                'start_pose': poses[0],
                'end_pose': poses[turns[0]['start_idx']]
            })
        
        # 添加转弯之间的直线段
        for i in range(len(turns) - 1):
            if turns[i+1]['start_idx'] > turns[i]['end_idx']:
                straight_segments.append({
                    'start_idx': turns[i]['end_idx'],
                    'end_idx': turns[i+1]['start_idx'],
                    'start_pose': poses[turns[i]['end_idx']],
                    'end_pose': poses[turns[i+1]['start_idx']]
                })
        
        # 添加最后一个转弯后的直线段
        if turns[-1]['end_idx'] < len(poses) - 1:
            straight_segments.append({
                'start_idx': turns[-1]['end_idx'],
                'end_idx': len(poses) - 1,
                'start_pose': poses[turns[-1]['end_idx']],
                'end_pose': poses[-1]
            })
    
    return turns, straight_segments

def merge_consecutive_turns(turns, max_gap=20, min_merged_length=10):
    """
    合并连续的小转弯成一个大转弯
    
    Args:
        turns: list of dict, 转弯段列表
        max_gap: 最大间隔（索引点数），小于此间隔的转弯将被合并
        min_merged_length: 合并后的最小长度
    
    Returns:
        merged_turns: list of dict, 合并后的转弯段
    """
    if len(turns) <= 1:
        return turns
    
    print(f"开始合并转弯，原始转弯数: {len(turns)}")
    
    merged_turns = []
    current_merged = turns[0].copy()
    
    for i in range(1, len(turns)):
        current_turn = turns[i]
        prev_turn = turns[i-1]
        
        # 检查两个转弯之间的间隔
        gap = current_turn['start_idx'] - prev_turn['end_idx']
        
        # 检查是否可以合并
        can_merge = False
        
        # 1. 间隔小于阈值
        if gap <= max_gap:
            # 2. 检查转弯方向是否一致（都是左转或都是右转）
            prev_angle = prev_turn['turn_angle']
            curr_angle = current_turn['turn_angle']
            
            print(f"检查转弯{i-1}和{i}合并: gap={gap}, prev_angle={np.degrees(prev_angle):.1f}°, curr_angle={np.degrees(curr_angle):.1f}°")
            
            # 如果两个转弯方向一致（同号），则可以合并
            if (prev_angle > 0 and curr_angle > 0) or (prev_angle < 0 and curr_angle < 0):
                can_merge = True
                print(f"  方向一致，可以合并")
            # 或者如果转弯角度都很小，也可以合并
            elif abs(prev_angle) < np.radians(10) and abs(curr_angle) < np.radians(10):
                can_merge = True
                print(f"  角度都很小，可以合并")
            else:
                print(f"  方向不一致且角度较大，不合并")
        else:
            print(f"转弯{i-1}和{i}间隔太大({gap} > {max_gap})，不合并")
        
        if can_merge:
            # 合并转弯
            current_merged['end_idx'] = current_turn['end_idx']
            current_merged['end_pose'] = current_turn['end_pose']
            
            # 更新合并后的统计信息
            start_idx = current_merged['start_idx']
            end_idx = current_merged['end_idx']
            
            # 重新计算合并后的统计信息
            current_merged['avg_heading_change'] = (current_merged['avg_heading_change'] + current_turn['avg_heading_change']) / 2
            current_merged['avg_curvature'] = (current_merged['avg_curvature'] + current_turn['avg_curvature']) / 2
            current_merged['max_curvature'] = max(current_merged['max_curvature'], current_turn['max_curvature'])
            current_merged['turn_angle'] = current_merged['turn_angle'] + current_turn['turn_angle']
            
            print(f"  合并转弯{i-1}和{i}，新长度: {end_idx - start_idx}")
            
        else:
            # 不能合并，结束当前合并段
            current_length = current_merged['end_idx'] - current_merged['start_idx']
            if current_length >= min_merged_length:
                merged_turns.append(current_merged)
                print(f"  结束合并段，长度: {current_length}")
            else:
                print(f"  合并段太短({current_length} < {min_merged_length})，丢弃")
            
            # 开始新的合并段
            current_merged = current_turn.copy()
    
    # 处理最后一个合并段
    current_length = current_merged['end_idx'] - current_merged['start_idx']
    if current_length >= min_merged_length:
        merged_turns.append(current_merged)
        print(f"  添加最后一个合并段，长度: {current_length}")
    else:
        print(f"  最后一个合并段太短({current_length} < {min_merged_length})，丢弃")
    
    print(f"合并完成，最终转弯数: {len(merged_turns)}")
    return merged_turns

def detect_and_merge_turns(poses, window_size=10, angle_threshold=15.0, curvature_threshold=0.005, 
                          min_turn_length=5, max_gap=20, min_merged_length=5):
    """
    检测转弯并合并连续的小转弯
    
    Args:
        poses: (N, 6) 车辆位姿
        window_size: 滑动窗口大小
        angle_threshold: 转弯角度阈值（度）
        curvature_threshold: 曲率阈值
        min_turn_length: 最小转弯段长度
        max_gap: 合并转弯的最大间隔
        min_merged_length: 合并后的最小长度
    
    Returns:
        merged_turns: list of dict, 合并后的转弯段
        straight_segments: list of dict, 直线段信息
        original_turns: list of dict, 原始转弯段（用于对比）
    """
    print(f"开始检测转弯，位姿点数: {len(poses)}")
    
    # 1. 检测原始转弯
    original_turns, _ = detect_turns_from_trajectory(
        poses, window_size, angle_threshold, curvature_threshold, min_turn_length, ignore_start_end_ratio=0.1
    )
    
    print(f"检测到原始转弯数: {len(original_turns)}")
    
    # 2. 合并连续转弯
    merged_turns = merge_consecutive_turns(original_turns, max_gap, min_merged_length)
    
    print(f"合并后转弯数: {len(merged_turns)}")
    
    # 3. 重新构建直线段
    straight_segments = []
    
    if not merged_turns:
        # 没有转弯，整个轨迹都是直线
        straight_segments.append({
            'start_idx': 0,
            'end_idx': len(poses) - 1,
            'start_pose': poses[0],
            'end_pose': poses[-1]
        })
        print("没有转弯，整个轨迹都是直线")
    else:
        # 添加起始直线段
        if merged_turns[0]['start_idx'] > 0:
            straight_segments.append({
                'start_idx': 0,
                'end_idx': merged_turns[0]['start_idx'],
                'start_pose': poses[0],
                'end_pose': poses[merged_turns[0]['start_idx']]
            })
            print(f"添加起始直线段: 0 -> {merged_turns[0]['start_idx']}")
        
        # 添加转弯之间的直线段
        for i in range(len(merged_turns) - 1):
            if merged_turns[i+1]['start_idx'] > merged_turns[i]['end_idx']:
                straight_segments.append({
                    'start_idx': merged_turns[i]['end_idx'],
                    'end_idx': merged_turns[i+1]['start_idx'],
                    'start_pose': poses[merged_turns[i]['end_idx']],
                    'end_pose': poses[merged_turns[i+1]['start_idx']]
                })
                print(f"添加转弯间直线段: {merged_turns[i]['end_idx']} -> {merged_turns[i+1]['start_idx']}")
        
        # 添加最后一个转弯后的直线段
        if merged_turns[-1]['end_idx'] < len(poses) - 1:
            straight_segments.append({
                'start_idx': merged_turns[-1]['end_idx'],
                'end_idx': len(poses) - 1,
                'start_pose': poses[merged_turns[-1]['end_idx']],
                'end_pose': poses[-1]
            })
            print(f"添加结尾直线段: {merged_turns[-1]['end_idx']} -> {len(poses) - 1}")
    
    print(f"构建了 {len(straight_segments)} 个直线段")
    return merged_turns, straight_segments, original_turns

def extract_straight_segments_points_and_poses(points, poses, road_segments, distance_threshold=5.0):
    """
    只保留所有直线段的点云和对应位姿
    Args:
        points: (N, 3) 全部点云
        poses: (M, 6) 全部位姿
        road_segments: list, analyze_road_structure_from_trajectory等返回的road_segments
        distance_threshold: 点云到直线的最大横向距离（米）
    Returns:
        straight_points_list: list，每个元素为一个直线段的点云 (K, 3)
        straight_poses_list: list，每个元素为一个直线段的位姿 (L, 6)
    """
    straight_points_list = []
    straight_poses_list = []
    
    print(f"开始提取直线段点云，总点数: {len(points)}, 道路段数: {len(road_segments)}")
    
    for i, seg in enumerate(road_segments):
        if seg['type'] != 'straight':
            continue
            
        print(f"处理直线段 {i}: {seg['start_idx']} -> {seg['end_idx']}")
        
        # 1. 取该直线段的起止位姿
        seg_poses = poses[seg['start_idx']:seg['end_idx']+1]
        straight_poses_list.append(seg_poses)
        
        # 2. 计算直线段方向
        start = seg['start_pose'][:2]
        end = seg['end_pose'][:2]
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 1e-3:
            print(f"  直线段 {i} 长度太短: {length:.6f}")
            continue
            
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])
        
        # 3. 选取属于该直线段的点云
        seg_points = []
        for pt in points:
            vec = pt[:2] - start
            proj = np.dot(vec, direction)
            lat_dist = np.abs(np.dot(vec, normal))
            if 0 <= proj <= length and lat_dist < distance_threshold:
                seg_points.append(pt)
        
        print(f"  直线段 {i}: 长度={length:.2f}m, 筛选到点数={len(seg_points)}")
        
        if len(seg_points) > 0:
            straight_points_list.append(np.array(seg_points))
        else:
            print(f"  直线段 {i}: 没有筛选到点云，可能distance_threshold太小")
    
    print(f"提取完成: {len(straight_points_list)} 个直线段有点云")
    return straight_points_list, straight_poses_list

def save_straight_segments(points_list, poses_list, output_dir="straight_segments"):
    """
    保存直线段点云为.pcd格式，位姿为TUM格式的txt文件，类别为.npy格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (pts, pss) in enumerate(zip(points_list, poses_list)):
        # 检查点云是否为空
        if len(pts) == 0:
            print(f"段 {i}: 点云为空，跳过保存")
            continue
            
        # 分离点云坐标和类别信息
        if pts.shape[1] == 4:  # x, y, z, class
            pts_xyz = pts[:, :3]  # 只取前3列作为坐标
            pts_class = pts[:, 3]  # 第4列作为类别
            has_class = True
        elif pts.shape[1] == 3:  # 只有x, y, z
            pts_xyz = pts
            pts_class = None
            has_class = False
        else:
            print(f"段 {i}: 点云格式不支持，期望3或4列，实际{pts.shape[1]}列")
            continue
            
        # 保存点云为.pcd格式
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_xyz)

            # 根据类别着色到RGB，便于后续通过颜色恢复类别
            if has_class:
                colors = np.zeros_like(pts_xyz)
                for idx, cls in enumerate(pts_class.astype(int)):
                    if cls == LANE_SOLID_CLASS:       # 绿色 - 实线
                        colors[idx] = [0.0, 1.0, 0.0]
                    elif cls == LANE_DASHED_CLASS:   # 红色 - 虚线
                        colors[idx] = [1.0, 0.0, 0.0]
                    elif cls == LANE_ARROW_CLASS:   # 黄色 - 箭头
                        colors[idx] = [1.0, 1.0, 0.0]
                    elif cls == STOP_LINE_CLASS:     # 蓝色 - 停止线
                        colors[idx] = [0.0, 0.0, 1.0]
                    else:                            # 其他/未知 - 灰色
                        colors[idx] = [0.6, 0.6, 0.6]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                # 无类别信息则统一浅灰
                pcd.colors = o3d.utility.Vector3dVector(np.full_like(pts_xyz, 0.8))

            pcd_filename = os.path.join(output_dir, f"lane_segment_{i:03d}.pcd")
            o3d.io.write_point_cloud(pcd_filename, pcd)
            
            # 如果有类别信息，额外保存为.npy，便于严格复原类别
            if has_class:
                class_filename = os.path.join(output_dir, f"lane_segment_{i:03d}_classes.npy")
                np.save(class_filename, pts_class)
                print(f"段 {i}: 点云点数={len(pts)}, 位姿点数={len(pss)}, 包含类别信息")
            else:
                print(f"段 {i}: 点云点数={len(pts)}, 位姿点数={len(pss)}")
                
        except Exception as e:
            print(f"段 {i}: 保存点云失败: {e}")
            continue
        
        # 保存位姿为TUM格式的txt文件
        try:
            poses_filename = os.path.join(output_dir, f"lane_segment_{i:03d}_poses.txt")
            with open(poses_filename, 'w') as f:
                # 写入TUM格式头部
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                # 写入位姿数据
                for idx, pose in enumerate(pss):
                    # pose格式: [x, y, z, roll, pitch, yaw] -> 需要转换为四元数
                    x, y, z = pose[0], pose[1], pose[2]
                    roll, pitch, yaw = pose[3], pose[4], pose[5]
                    
                    # 欧拉角转四元数
                    cy = np.cos(yaw * 0.5)
                    sy = np.sin(yaw * 0.5)
                    cp = np.cos(pitch * 0.5)
                    sp = np.sin(pitch * 0.5)
                    cr = np.cos(roll * 0.5)
                    sr = np.sin(roll * 0.5)
                    
                    qw = cr * cp * cy + sr * sp * sy
                    qx = sr * cp * cy - cr * sp * sy
                    qy = cr * sp * cy + sr * cp * sy
                    qz = cr * cp * sy - sr * sp * cy
                    
                    # 使用递增时间戳
                    timestamp = idx * 0.1  # 每帧间隔0.1秒
                    
                    f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        except Exception as e:
            print(f"段 {i}: 保存位姿失败: {e}")
            continue
    
    print(f"已保存{len(points_list)}段直线段到 {output_dir}/")
    if has_class:
        print(f"文件格式: lane_segment_XXX.pcd (点云), lane_segment_XXX_poses.txt (TUM格式位姿), lane_segment_XXX_classes.npy (类别)")
    else:
        print(f"文件格式: lane_segment_XXX.pcd (点云), lane_segment_XXX_poses.txt (TUM格式位姿)")

def main():
    """主函数示例"""
    # 示例用法
    print("道路结构分析器")
    print("=" * 50)
    
    # 这里可以添加具体的测试代码
    print("请使用 analyze_and_segment_road_lanes() 函数来分析道路结构")
    # 读取点云和位姿
    pcd_file = "segments_arrow_temp/lane_segment_006.pcd"
    pose_file = "segments_arrow_temp/lane_segment_006_poses.txt"
    
    points = load_point_cloud(pcd_file)
    poses = load_poses(pose_file)
    # 车道线分析
    road_analysis = analyze_and_segment_road_lanes(points, poses, dashed_class=LANE_DASHED_CLASS, solid_class=LANE_SOLID_CLASS, visualize=True)
    road_segments = road_analysis['road_structure']['road_segments']
    # 提取直线段点云和位姿
    straight_points_list, straight_poses_list = extract_straight_segments_points_and_poses(points, poses, road_segments, distance_threshold=20.0)
    print(f"共提取到{len(straight_points_list)}段直线段点云")
    # 保存为文件
    save_straight_segments(straight_points_list, straight_poses_list)

if __name__ == "__main__":
    main() 