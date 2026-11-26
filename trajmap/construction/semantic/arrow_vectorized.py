"""Research inspection: yellow-point DBSCAN clusters and GUI bounding boxes.

This does not export paper arrow rectangles, map attributes, or junction graphs.
"""

import numpy as np

# 定义类别ID
LANE_SOLID_CLASS = 1
LANE_DASHED_CLASS = 2
LANE_ARROW_CLASS = 3
STOP_LINE_CLASS = 5

def filter_points_by_class(points, class_ids):
    """根据类别ID过滤点云"""
    if len(points) == 0:
        return np.array([])
    mask = np.isin(points[:, 3], class_ids)
    return points[mask]

def load_point_cloud(filepath):
    """加载点云文件"""
    import open3d as o3d

    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # 根据颜色判断类别
        class_ids = np.zeros(len(points), dtype=int)
        for i, color in enumerate(colors):
            if np.allclose(color, [1, 1, 0]):  # 绿色 - 箭头线
                class_ids[i] = LANE_ARROW_CLASS
        
        # 合并点云坐标和类别
        points_with_class = np.column_stack((points, class_ids))
        return filter_points_by_class(points_with_class, [LANE_ARROW_CLASS])
    except Exception as e:
        print(f"加载点云文件失败: {e}")
        return None

def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Semantic RGB PCD; opens an Open3D GUI")
    args = parser.parse_args(argv)
    if not Path(args.input).is_file():
        parser.error("Input cloud does not exist: " + args.input)
    import open3d as o3d

    cloud_path = args.input
    point_data = load_point_cloud(cloud_path)
    
    if point_data is None:
        print("无法加载点云数据，程序退出。")
        return
    
    print(f"点云数据形状: {point_data.shape}")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data[:, :3])  # 使用3D坐标
    
    # 使用DBSCAN聚类算法
    eps = 1.2  # 聚类半径
    min_points = 50  # 最小点数
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    # 获取聚类结果
    unique_labels = np.unique(labels)
    geometries = []  # 用于存储所有几何对象（点云和边界框）
    
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
        # 提取当前聚类的点
        cluster_points = point_data[labels == label]
        
        # 创建点云对象用于当前聚类
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points[:, :3])
        
        # 为聚类点设置颜色（可选，方便可视化区分）
        pcd_cluster.paint_uniform_color(np.random.rand(3))  # 随机颜色
        
        # 计算边界框
        min_bound = np.min(cluster_points[:, :3], axis=0)
        max_bound = np.max(cluster_points[:, :3], axis=0)
        
        # 创建轴对齐边界框
        # 使用 create_from_axis_aligned_bounding_box 需要中心和范围
        center = (min_bound + max_bound) / 2
        extent = max_bound - min_bound
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        # 设置边界框颜色（例如红色）
        box.color = [1, 0, 0]
        
        # 将点云和边界框添加到几何对象列表
        geometries.append(pcd_cluster)
        geometries.append(box)
    
    # 可视化所有点云和边界框
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with Bounding Boxes")

if __name__ == "__main__":
    main()
