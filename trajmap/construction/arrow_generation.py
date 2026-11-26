"""Headless extraction of the approved media arrow construction implementation.

Source: /media/oymj/F/HDMap-main/process_lane_segments.py
SHA256: 126fe7076c9b1bbb97f22586840a72e6b7dd134076516ee00fc50b1c15f64b6f
Active definitions: deal_arrow_process (3502), bbox/PCA/group helpers
(4672-4859), last boundary helpers (4860-4973), processing (4974-5245).
Earlier shadowed boundary functions are deliberately NOT used.

Retains Open3D XYZ DBSCAN (0.9, 800), the optional <=0.5m point filter,
the <=2m distance to infinite supporting lines, PCA grouping at 15 degrees,
and the original two-sided 5m test. Boxes follow correct_direction, not PCA;
minimum length/width is 1m, with no maximum. Secondary DBSCAN (0.6, 100)
for width >2m or length >10m changes dimensions ONLY: its box midpoint and
Z mean remain those computed BEFORE reclustering. This stale-center behavior
is intentional source compatibility, not a recentered core box. arrow_info
center is separately the original cluster mean, never the box midpoint.

No GUI, file IO, external project imports, or topology inference. Open3D is
loaded only when clustering. Optional solid/dashed inputs and segment_id are
kept for caller compatibility but were used only for removed diagnostics.
Visualization helper return slots are None (bbox) or [] (processing).

Input adaptation flattens nested window arrays in order; nested ndarray
windows retain their first two vertices, as in the original list branch.
Compatibility is NOT blanket equivalence: the source's last boundary filter
skips list inputs flattened to fewer than FOUR points, even though two points
form a valid line. Here a single two-point window is accepted as a normalized
polyline, so it can admit arrows that the source skipped. Deeper nesting and
plain coordinate lists that previously failed parsing can also be accepted.
Numerical parity applies to valid flat arrays and supported endpoint-window
lists with at least four flattened points, not to those source parser failures.
Empty/degenerate lanes are ignored;
zero-length edges are skipped when finding a local normal. Valid calls always
return lists, including empty/no-lane/all-filtered cases. Malformed required
inputs, nonfinite XYZ/XY, zero XY direction and unsupported fitting_type raise
ValueError rather than returning raw point arrays or producing invalid boxes.
"""
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.decomposition import PCA


def _arrow_points(points):
    try:
        points = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("arrow_lane_points must be a numeric N x >=3 array") from exc
    if points.shape == (0,):
        return np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise ValueError("arrow_lane_points must contain finite XYZ in an N x >=3 array")
    return points


def _direction(direction):
    try:
        direction = np.asarray(direction, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correct_direction must be a finite vector with nonzero XY") from exc
    if (direction.ndim != 1 or direction.size < 2 or not np.isfinite(direction).all()
            or not np.isfinite(np.linalg.norm(direction[:2]))
            or np.linalg.norm(direction[:2]) == 0):
        raise ValueError("correct_direction must be a finite vector with nonzero XY")
    return direction


def _lane_points(value, nested=False):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("lane geometry must contain coordinate arrays")
    if len(value) == 0:
        return []
    if all(np.isscalar(item) for item in value):
        point = np.asarray(value, dtype=float)
        if point.size < 2 or not np.isfinite(point).all():
            raise ValueError("lane coordinates must have finite XY")
        return [point[:2]]
    if nested and isinstance(value, np.ndarray) and value.ndim == 2:
        value = value[:2]
    points = []
    for child in value:
        points.extend(_lane_points(child, nested=True))
    return points


def _normalise_lanes(lane_segments):
    lanes = []
    if lane_segments is None:
        return lanes
    for entry in lane_segments:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("lane_segments entries must be (geometry, lane_type, lane_num)")
        geometry, lane_type, lane_num = entry
        points = np.asarray(_lane_points(geometry), dtype=float)
        if len(points) < 2 or not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0):
            continue
        lanes.append((points, lane_type, lane_num))
    return lanes


def _cluster_labels(points, eps, min_points):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))


def get_max_bbox_size_from_clusters(arrow_clusters):
    max_size = [0, 0, 0]
    for cluster_points, _ in arrow_clusters:
        points = cluster_points[:, :3]
        size = np.max(points, axis=0) - np.min(points, axis=0)
        for i in range(3):
            max_size[i] = max(max_size[i], size[i])
    return max_size


def create_standardized_bbox(cluster_points, max_size, direction):
    """Return (None, None, original_box_center, four_XYZ_corners).

    max_size affected only the removed display OBB's height. Secondary DBSCAN
    retains the pre-clustering midpoint and Z mean, exactly as in the source.
    """
    points_2d = cluster_points[:, :2]
    direction_2d = direction[:2] / (np.linalg.norm(direction[:2]) + 1e-8)
    perp_direction = np.array([-direction_2d[1], direction_2d[0]])
    proj_along = points_2d @ direction_2d
    proj_perp = points_2d @ perp_direction
    min_along, max_along = np.min(proj_along), np.max(proj_along)
    min_perp, max_perp = np.min(proj_perp), np.max(proj_perp)
    center_along = (min_along + max_along) / 2
    center_perp = (min_perp + max_perp) / 2
    center_2d = center_along * direction_2d + center_perp * perp_direction
    z_mean = np.mean(cluster_points[:, 2])
    center = np.array([center_2d[0], center_2d[1], z_mean])
    actual_length = max_along - min_along
    actual_width = max_perp - min_perp

    if actual_width > 2.0 or actual_length > 10.0:
        labels = _cluster_labels(cluster_points, eps=0.6, min_points=100)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) > 0:
            largest_cluster_label = unique_labels[np.argmax(counts)]
            core_points = cluster_points[labels == largest_cluster_label]
            if len(core_points) > 10:
                core_proj_along = core_points[:, :2] @ direction_2d
                core_proj_perp = core_points[:, :2] @ perp_direction
                min_along, max_along = np.min(core_proj_along), np.max(core_proj_along)
                min_perp, max_perp = np.min(core_proj_perp), np.max(core_proj_perp)
                actual_length = max_along - min_along
                actual_width = max_perp - min_perp

    extent_along = max(1.0, actual_length) / 2.0
    extent_perp = max(1.0, actual_width) / 2.0
    corners = np.array([
        center_2d + extent_along * direction_2d + extent_perp * perp_direction,
        center_2d + extent_along * direction_2d - extent_perp * perp_direction,
        center_2d - extent_along * direction_2d - extent_perp * perp_direction,
        center_2d - extent_along * direction_2d + extent_perp * perp_direction,
    ])
    corners_3d = np.column_stack((corners, np.full(4, z_mean)))
    return None, None, center, corners_3d


def get_cluster_direction(cluster_points):
    if len(cluster_points) < 3:
        return np.array([1, 0, 0])
    pca = PCA(n_components=3)
    pca.fit(cluster_points[:, :3])
    return pca.components_[0]


def group_clusters_by_direction(clusters, angle_threshold_deg=15.0):
    if not clusters:
        return []
    directions = [get_cluster_direction(points) for points, _ in clusters]
    groups = []
    remaining_indices = list(range(len(clusters)))
    while remaining_indices:
        current_index = remaining_indices.pop(0)
        current_direction = directions[current_index]
        new_group = {"clusters": [clusters[current_index]], "avg_direction": current_direction}
        indices_to_remove = []
        for i in remaining_indices:
            dot_product = np.clip(np.dot(current_direction, directions[i]), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180.0 / np.pi
            if angle < angle_threshold_deg or abs(angle - 180.0) < angle_threshold_deg:
                new_group["clusters"].append(clusters[i])
                indices_to_remove.append(i)
        for i in sorted(indices_to_remove, reverse=True):
            if i in remaining_indices:
                remaining_indices.remove(i)
        groups.append(new_group)
    return groups


def get_line_direction(line, point):
    proj_dist = line.project(point)
    coords = np.array(line.coords)
    total_length = 0
    for i in range(len(coords) - 1):
        segment = LineString(coords[i:i + 2])
        seg_length = segment.length
        if seg_length == 0:
            continue
        if total_length <= proj_dist <= total_length + seg_length:
            direction = coords[i + 1] - coords[i]
            direction = direction / np.linalg.norm(direction)
            return np.array([-direction[1], direction[0]])
        total_length += seg_length
    return np.array([0, 1])


def check_both_sides(line, point, all_lines, distance=5.0):
    normal = get_line_direction(line, point)
    p = np.array([point.x, point.y])
    check_point_pos = p + distance * normal
    check_point_neg = p - distance * normal
    pos_point, neg_point = Point(check_point_pos), Point(check_point_neg)
    pos_has_line, neg_has_line = False, False
    for other_line in all_lines:
        if other_line.distance(pos_point) <= distance:
            pos_has_line = True
        if other_line.distance(neg_point) <= distance:
            neg_has_line = True
        if pos_has_line and neg_has_line:
            break
    return pos_has_line and neg_has_line, check_point_pos, check_point_neg


def is_arrow_in_lane_boundaries(cluster_center, lane_segments):
    """Last source definition: <=2m to ANY supporting line, not a lane polygon.

    lane_segments here are normalized numeric polylines, as prepared by the
    public entry point. The projection is intentionally not clamped to an edge.
    """
    for seg_line, _, _ in lane_segments:
        for i in range(len(seg_line) - 1):
            p1, p2 = seg_line[i, :2], seg_line[i + 1, :2]
            line_vec = p2 - p1
            norm = np.linalg.norm(line_vec)
            if norm == 0:
                continue
            line_vec = line_vec / norm
            perpendicular = np.array([-line_vec[1], line_vec[0]])
            point_vec = np.array(cluster_center[:2]) - p1
            distance = np.abs(np.dot(point_vec, perpendicular))
            if distance <= 2.0:
                return True
    return False


def filter_arrow_clusters_by_boundaries(arrow_clusters, lane_segments):
    filtered_clusters = []
    for cluster_points, label in arrow_clusters:
        cluster_center = np.mean(cluster_points[:, :3], axis=0)
        if is_arrow_in_lane_boundaries(cluster_center, lane_segments):
            filtered_clusters.append((cluster_points, label))
    return filtered_clusters


def process_arrows_with_lane_filtering(arrow_lane_points, solid_lane_points,
                                      improved_dashed_line, lane_segments, correct_direction):
    """Return (empty_visualizations, valid_clusters, arrow_data); lanes normalized."""
    if len(arrow_lane_points) == 0 or not lane_segments:
        return [], [], []
    labels = _cluster_labels(arrow_lane_points, eps=0.9, min_points=800)
    arrow_clusters = [(arrow_lane_points[labels == label], label)
                      for label in np.unique(labels) if label != -1]
    filtered_clusters = filter_arrow_clusters_by_boundaries(arrow_clusters, lane_segments)
    lane_linestrings = [LineString(line[:, :2]) for line, _, _ in lane_segments]
    arrow_data, valid_clusters = [], []
    for group_idx, group in enumerate(group_clusters_by_direction(filtered_clusters)):
        group_max_size = get_max_bbox_size_from_clusters(group["clusters"])
        for cluster_points, label in group["clusters"]:
            _, _, bbox_center, bbox_vertices = create_standardized_bbox(
                cluster_points, group_max_size, correct_direction)
            cluster_center = np.mean(cluster_points[:, :3], axis=0)
            center_point = Point(cluster_center[:2])
            min_dist, nearest_line = float("inf"), None
            for line in lane_linestrings:
                projection = line.interpolate(line.project(center_point))
                dist = center_point.distance(projection)
                if dist < min_dist:
                    min_dist, nearest_line = dist, line
            is_valid, _, _ = check_both_sides(nearest_line, center_point, lane_linestrings, distance=5.0)
            if is_valid:
                valid_clusters.append((cluster_points, label))
                arrow_data.append({"bbox_vertices": bbox_vertices, "center": cluster_center,
                                   "label": label, "group_id": group_idx})
    return [], valid_clusters, arrow_data


def deal_arrow_process(lane_segments, arrow_lane_points, segment_id, correct_direction,
                       solid_lane_points=None, improved_dashed_line=None, fitting_type=1):
    """Return arrow_info dictionaries; invalid required inputs raise ValueError.

    arrow_lane_points is numeric N x >=3 (XYZ first), direction has nonzero XY,
    and lane_segments contains (polyline/window_arrays, lane_type, lane_num).
    fitting_type=1 removes points <=0.5m from finite lane polylines; 0 bypasses
    only that stage. Both modes retain cluster and two-sided boundary checks.
    """
    if fitting_type not in (0, 1):
        raise ValueError("fitting_type must be 0 or 1")
    points = _arrow_points(arrow_lane_points)
    if len(points) == 0:
        return []
    direction = _direction(correct_direction)
    lanes = _normalise_lanes(lane_segments)
    if not lanes:
        return []
    if fitting_type == 1:
        lines = [LineString(line[:, :2]) for line, _, _ in lanes]
        true_arrows_mask = np.ones(len(points), dtype=bool)
        for i, p_xy in enumerate(points[:, :2]):
            point = Point(p_xy)
            min_dist = float("inf")
            for line in lines:
                projection = line.interpolate(line.project(point))
                dist = point.distance(projection)
                if dist < min_dist:
                    min_dist = dist
            if min_dist <= 0.5:
                true_arrows_mask[i] = False
        points = points[true_arrows_mask]
    if len(points) == 0:
        return []
    _, _, arrow_data = process_arrows_with_lane_filtering(
        points, solid_lane_points, improved_dashed_line, lanes, direction)
    return arrow_data
