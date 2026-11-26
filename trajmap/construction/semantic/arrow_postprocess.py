"""Original arrow-mask recovery; the caller supplies the dataset car mask."""

import numpy as np
import math
def line_to_kb(line):
    x1, y1, x2, y2 = line
    if x2 == x1:
        k = np.inf
        b = x1
    else:
        k = (y2 - y1) / (x2 - x1 + 1e-6)
        b = y1 - k * x1
    return k, b

def has_infinity(lst):
    return any(math.isinf(x) for x in lst if isinstance(x, float))
def cluster_lines(lines, eps_k=0.1, eps_b=30):
    """
    lines: N x 4 array, 每行[x1, y1, x2, y2]
    eps_k: 斜率聚类阈值
    eps_b: 截距聚类阈值
    返回：每个聚类的代表直线
    """
    from sklearn.cluster import DBSCAN

    if lines is None or len(lines) == 0:
        return []
    kb_list = [line_to_kb(l[0]) for l in lines]
    kb_arr = np.array(kb_list)
    # 归一化斜率和截距
    k_arr = kb_arr[:,0]
    b_arr = kb_arr[:,1]
    # 构造特征向量
    X = np.stack([k_arr / eps_k, b_arr / eps_b], axis=1)  # 归一化后，欧氏距离<1即为同类
    for x in X:
        if has_infinity(x):
            return []
    print("x is ", X)
    db = DBSCAN(eps=1.0, min_samples=1).fit(X)
    labels = db.labels_
    merged_lines = []
    for label in set(labels):
        idxs = np.where(labels == label)[0]
        # 取该类中最长的直线作为代表
        max_len = 0
        best_line = None
        for idx in idxs:
            x1, y1, x2, y2 = lines[idx][0]
            length = np.hypot(x2-x1, y2-y1)
            if length > max_len:
                max_len = length
                best_line = [x1, y1, x2, y2]
        merged_lines.append(best_line)
    return merged_lines

def get_lane_mask(seg_mask, lane_class=1):
    return (seg_mask == lane_class).astype(np.uint8) * 255

def get_lane_lines(lane_mask, min_line_length=50, max_line_gap=20):
    import cv2

    lines = cv2.HoughLinesP(
        lane_mask,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # for i in range(len(lines)):
    #     print(lines[i], lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
    #     cv2.line(rimg, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255,0,255),5)
    # cv2.imshow("image", rimg)
    # cv2.waitKey()
    return lines  # lines: [ [x1, y1, x2, y2], ... ]

def compute_intersections(lines):
    """
    lines: N x 4 array, 每行[x1, y1, x2, y2]
    返回：所有交点的列表
    """
    intersections = []
    if lines is None or len(lines) < 2:
        return intersections
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            l1 = lines[i][0]
            l2 = lines[j][0]
            pt = compute_line_intersection(l1, l2)
            if pt is not None:
                intersections.append(pt)
    return intersections

def compute_line_intersection(l1, l2):
    """
    l1, l2: [x1, y1, x2, y2]
    返回：交点(x, y)或None
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

def estimate_vanishing_point(intersections, img_shape):
    """
    intersections: 交点列表
    img_shape: (h, w)
    返回：vanishing point (x, y)
    """
    if not intersections:
        # 默认图像中心上方1/3
        h, w = img_shape
        return (w//2, h//3)
    pts = np.array(intersections)
    # 用2D直方图统计最密集区域
    h, w = img_shape
    hist, xedges, yedges = np.histogram2d(pts[:,0], pts[:,1], bins=[w//20, h//20])
    idx = np.unravel_index(np.argmax(hist), hist.shape)
    x_center = int((xedges[idx[0]] + xedges[idx[0]+1]) / 2)
    y_center = int((yedges[idx[1]] + yedges[idx[1]+1]) / 2)
    return (x_center, y_center)

def get_vanishing_point_from_segmentation(seg_mask, lane_class=1):
    lane_mask = get_lane_mask(seg_mask, lane_class)
    lines = get_lane_lines(lane_mask)
    merged_lines = cluster_lines(lines)
    # print("merged_lines shape is ", np.array(merged_lines).shape)
    # rimg = cv2.imread(img_path)
    # for l in merged_lines:
    #     cv2.line(rimg, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 5)
    # cv2.imshow("img", rimg)
    # cv2.waitKey()
    intersections = compute_intersections([[l] for l in merged_lines])
    vp = estimate_vanishing_point(intersections, seg_mask.shape)
    return vp

def extract_arrows_from_segmentation(result_image, arrow_class=3, min_area=50, angle_threshold=15, center_bias=0.3, *, car_mask):
    """
    输入: result_image（inferencer分割结果，dict，含'predictions'）
    返回: arrows, noise_arrows
    arrows: [{'mask': mask, 'center': (x, y), 'angle': angle, 'area': area}]
    noise_arrows: 同上
    """
    import cv2

    car_mask = np.asarray(car_mask)
    if car_mask.shape != result_image['predictions'].shape:
        raise ValueError("car_mask must match the semantic prediction shape")
    candidate_classes = [1, 2]

    seg_mask = result_image['predictions'].astype(np.uint8)
    seg_mask = seg_mask * car_mask
    h, w = seg_mask.shape
    # 1. 提取箭头类别区域
    arrow_mask = (seg_mask == arrow_class).astype(np.uint8)

    vp = get_vanishing_point_from_segmentation(seg_mask)
    print(vp)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arrow_mask, connectivity=4)
    recovered_arrows = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        if mask.sum() < min_area:
            continue
        # 箭头中心
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = np.array([cx, cy])
        # 灭点方向
        if vp is None:
            # 默认图像中心上方1/3
            vp = (w // 2, h // 3)
        v_dir = np.array(vp) - center
        v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-6)

        # 区域生长
        grown = mask.copy()
        for _ in range(100):
            # 膨胀
            dilated = cv2.dilate(grown, np.ones((15, 15), np.uint8), iterations=1)
            # 找到新扩展的边界
            border = (dilated - grown) > 0
            # 只吸收可疑类别
            absorb = np.zeros_like(seg_mask, dtype=bool)
            for c in candidate_classes:
                # absorb = absorb | ((seg_mask == c) & border)
                candidate = ((seg_mask == c) & border)
                y_idx, x_idx = np.where(candidate)
                rel = np.stack([x_idx - center[0], y_idx - center[1]], axis=1)
                rel_norm = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6)
                # 计算与v_dir的夹角
                cos_angle = np.clip(rel_norm @ v_dir, -1, 1)
                angle = np.arccos(np.abs(cos_angle)) * 180 / np.pi  # 允许正向和反向
                valid = angle < 10
                absorb[y_idx[valid], x_idx[valid]] = True

            if not np.any(absorb):
                break
            grown[absorb] = 1

        recovered_arrows.append(grown)
    # 合并所有箭头区域
    recovered_arrow_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for region in recovered_arrows:
        recovered_arrow_mask = np.logical_or(recovered_arrow_mask, region)
    recovered_arrow_mask = recovered_arrow_mask.astype(np.uint8)

    # cv2.circle(recovered_arrow_mask, vp, 10, (0, 0, 255), -1)
    # cv2.imshow('Vanishing Point', vis)
    # cv2.imshow("img", recovered_arrow_mask*50)
    # cv2.waitKey()

    return recovered_arrow_mask
