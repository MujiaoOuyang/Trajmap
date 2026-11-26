"""Original R-tree lane-number and label calculations.

Requires EPSG:32650 2D polylines with >=2 vertices and >=7 ordered fields,
id/Id at index 3, and existing LaneNumber/Label fields. Index 5 is the
unchanged skip gate (nonzero skips); index 6 is the unchanged angle gate.
Values are compared to numeric zero exactly as in the original; string
"0" is NOT numeric zero. Callers must supply the original positional schema.
Lengths 20/25/30 and angle thresholds 10/60 retain the source's metre-based
assumptions. Multipart geometry is flattened on output as in the original.
This is a separate historical variant, not an automatic second pipeline stage.
"""
import math
from contextlib import ExitStack

if __package__:
    from ._shapefile_io import attribute_parser, check_output, read_crs, write_crs, validate_attributes
else:
    from _shapefile_io import attribute_parser, check_output, read_crs, write_crs, validate_attributes

ZERO = 1e-9

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

def normalize_angle(angle):
    """将角度规范到 [-180, 180)"""
    while angle >= 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def angle_diff(a1, a2):
    diff = abs(normalize_angle(a1 - a2))
    return min(diff, 360 - diff)

def rotate90(dx, dy):
    """顺时针90度旋转"""
    return dy, -dx

def is_segment_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def midpoint(points):
    pts = [Point(p[0], p[1]) for p in points]
    mx = sum(p.x for p in pts) / len(pts)
    my = sum(p.y for p in pts) / len(pts)
    return Point(mx, my), pts[0], pts[-1]

def perpendicular_line(mid, start, end, length=20.0):
    """构造长度为 length 的垂直线（双向）"""
    dx = end.x - start.x
    dy = end.y - start.y
    dist = math.hypot(dx, dy)
    if dist < ZERO:
        return None, None

    # 单位法向量（顺时针90度）
    nx, ny = rotate90(dx / dist, dy / dist)
    pA = Point(mid.x + nx * length, mid.y + ny * length)
    pB = Point(mid.x - nx * length, mid.y - ny * length)
    return pA, pB


def process(input_shp, output_shp, *, input_crs=None):
    import shapefile
    from pyproj import CRS
    from rtree import index

    output_shp = check_output(output_shp, (input_shp,))
    crs = read_crs(input_shp, input_crs)
    if crs != CRS.from_epsg(32650):
        raise ValueError("Original lane-number calculations require EPSG:32650.")
    with ExitStack() as stack:
        reader = stack.enter_context(shapefile.Reader(str(input_shp)))
        validate_attributes(reader)
        shapes = reader.shapes()
        records = reader.records()
        fields = reader.fields[1:]  # 跳过 DeletionFlag

        # 预计算：角度、边界框、中点、FID
        n = len(shapes)
        angles = []
        bboxes = []
        midpoints = []
        fids = []
        start_points = []
        end_points = []

        for i in range(n):
            pts = shapes[i].points
            if len(pts) < 2:
                continue
            p0, p1 = Point(pts[0][0], pts[0][1]), Point(pts[1][0], pts[1][1])
            dx, dy = p1.x - p0.x, p1.y - p0.y
            angle = math.atan2(dy, dx) * 180 / math.pi
            mid, sp, ep = midpoint(pts)
            bbox = shapes[i].bbox  # [xmin, ymin, xmax, ymax]

            angles.append(angle)
            bboxes.append(bbox)
            midpoints.append(mid)
            start_points.append(sp)
            end_points.append(ep)
            fids.append(records[i][3])  # FID 在第3列（0-based）

        # 构建 R-tree 空间索引
        idx = index.Index()
        for i in range(n):
            bbox = bboxes[i]
            idx.insert(i, bbox)

        # 输出
        w = stack.enter_context(shapefile.Writer(str(output_shp), shapeType=3))  # POLYLINE
        for field in fields:
            w.field(*field)

        # 新字段（可选）
        # w.field('LaneNumber', 'N', 10, 0)
        # w.field('Label', 'N', 10, 0)

        for i in range(n):
            fid = fids[i]

            mid = midpoints[i]
            sp, ep = start_points[i], end_points[i]
            angle_i = angles[i]
            record_i = records[i]

            if records[i][5]!=0:
                new_record = []
                for k, field_name in enumerate([f[0] for f in fields]):
                    new_record.append(record_i[k])
                w.record(*new_record)
                w.line([shapes[i].points])
                continue
            print(records[i][5])

            # 构造垂直线
            pA, pB = perpendicular_line(mid, sp, ep, length=25.0)  # 稍长一点防漏
            if pA is None:
                lane_num, label = 1, 1
            else:
                # 搜索附近线段
                # --- 在主循环中处理第 i 条线 ---
                candidates = list(idx.intersection((mid.x - 30, mid.y - 30, mid.x + 30, mid.y + 30)))
                parallel_lines = []

                for j in candidates:
                    if j == i or fids[j] == fid:
                        continue
                    if angle_diff(angles[i], angles[j]) > (60 if records[j][6] != 0 else 10):
                        continue
                    pts = shapes[j].points
                    if len(pts) < 2: continue
                    c = Point(pts[0][0], pts[0][1])
                    d = Point(pts[-1][0], pts[-1][1])
                    if is_segment_intersect(pA, pB, c, d):
                        parallel_lines.append(j)


                # === 关键修复：从右到左排序 ===
                def cross(j):
                    sp, ep = start_points[i], end_points[i]
                    dx_s = ep.x - sp.x
                    dy_s = ep.y - sp.y
                    dx_t = midpoints[j].x - midpoints[i].x
                    dy_t = midpoints[j].y - midpoints[i].y
                    return dx_s * dy_t - dy_s * dx_t


                parallel_lines.sort(key=cross)  # cross 小的（负的）在前 → 右侧在前

                # === 计算 Label：右侧有几条？===
                label = 1
                for j in parallel_lines:
                    if cross(j) < -ZERO:  # 右侧
                        label += 1
                    else:
                        break

                lane_num = len(parallel_lines) + 1

            # 写入
            new_record = []
            for k, field_name in enumerate([f[0] for f in fields]):
                if field_name == 'LaneNumber':
                    new_record.append(lane_num)
                elif field_name == 'Label':
                    new_record.append(label)
                else:
                    new_record.append(record_i[k])
            w.record(*new_record)
            w.line([shapes[i].points])

        w.close()

        write_crs(output_shp, crs)
        print("Saved lane number and label:", output_shp)
        return output_shp


def main(argv=None):
    args = attribute_parser(__doc__).parse_args(argv)
    process(args.input, args.output, input_crs=args.input_crs)


if __name__ == "__main__":
    main()
