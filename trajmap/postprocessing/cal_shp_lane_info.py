"""Original lane-count, label, width and arrow-center calculations.

Schema: type[0], center[2], id/Id[3], angle gate[6]; existing Label,
LaneNumber and LaneWidth fields are required. Gate values are compared to
numeric zero exactly as in the original, without coercing string fields.
Only the first two vertices drive the width calculation; vertical first
segments use infinite slope and a horizontal normal. Geometry must
be 2D polylines. Geographic inputs are transformed to EPSG:32650; projected
inputs must already be EPSG:32650. Arrow centers must always contain WGS84
lon,lat[,suffix], even when geometry is projected. Other centers are retained.
No field reordering, segmentation, or threshold/count changes are performed.
"""
import math
from contextlib import ExitStack

if __package__:
    from ._shapefile_io import attribute_parser, check_output, read_crs, write_crs, validate_attributes
else:
    from _shapefile_io import attribute_parser, check_output, read_crs, write_crs, validate_attributes

ZERO = 1e-9

class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

class Vector(object):
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y

def negative(vector):
    return Vector(vector.end, vector.start)

def vector_product(vectorA, vectorB):
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y

def is_intersected(A, B, C, D):
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)

    return (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
        and (vector_product(CA, CB) * vector_product(DA, DB) <= ZERO)

def get_point_by_offset(midP, endP, offset):
    deltaL = math.sqrt(math.pow(endP.x - midP.x, 2) + math.pow(endP.y - midP.y, 2))
    ratio = offset / deltaL
    px = midP.x + ratio * (endP.x - midP.x)
    py = midP.y + ratio * (endP.y - midP.y)
    return Point(px, py)

def parallel_line_distance(p1, p2, q1, q2):
    dx1 = p2.x - p1.x
    dy1 = p2.y - p1.y
    nx, ny = -dy1, dx1

    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2
    qx = (q1.x + q2.x) / 2
    qy = (q1.y + q2.y) / 2

    vx = qx - mx
    vy = qy - my

    norm = math.hypot(nx, ny)
    if norm < 1e-9:
        return 0.0
    distance = abs(vx * nx + vy * ny) / norm
    return distance


def process(input_shp, outshp, *, input_crs=None):
    import shapefile
    from pyproj import Transformer, CRS

    outshp = check_output(outshp, (input_shp,))
    crs = read_crs(input_shp, input_crs)
    if not crs.is_geographic and crs != CRS.from_epsg(32650):
        raise ValueError("Original attribute calculations require geographic CRS or EPSG:32650.")
    with ExitStack() as stack:
        file = stack.enter_context(shapefile.Reader(str(input_shp)))
        validate_attributes(file, width=True)
        records = file.shapeRecords()
        # 将 shapes 和 records 绑定在一起，方便后续在内存中直接修改和读取
        border = [r.shape for r in records]

        needs_transform = crs.is_geographic
        transformer = Transformer.from_crs(crs, "epsg:32650", always_xy=True) if needs_transform else None

        # ================= 2. 内存级别的数据预处理转换 =================
        # 专门为 arrow 的 center 准备的转换器
        arrow_center_transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)

        for r in records:
            lane_type = r.record[0]

            # 1. 转换线段节点坐标 (仅当地图本身是经纬度时才执行)
            if needs_transform:
                xs, ys = [], []
                for idx in range(len(r.shape.points)):
                    pt = r.shape.points[idx]
                    lon, lat = pt[0], pt[1]
                    x, y = transformer.transform(lon, lat)
                    r.shape.points[idx] = [x, y]
                    xs.append(x)
                    ys.append(y)

                # 必须同步更新每个要素的 Bbox
                if xs and ys:
                    r.shape.bbox = [min(xs), min(ys), max(xs), max(ys)]

            # 2. 【核心修改】仅针对 type == 'arrow' 的要素转换 center
            # 无论底图是什么坐标系，都强制把它从经纬度转成 UTM；如果不是 arrow，则什么都不做
            if lane_type == 'arrow':
                center_val = r.record[2]
                if isinstance(center_val, str) and ',' in center_val:
                    parts = center_val.split(',')
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        x, y = arrow_center_transformer.transform(lon, lat)
                        new_parts = [f"{x:.6f}", f"{y:.6f}"]
                        # 如果有第三个参数（比如方向角），原样拼接在后面
                        if len(parts) > 2:
                            new_parts.extend(parts[2:])
                        r.record[2] = ','.join(new_parts)
                    except ValueError:
                        pass


        # ================= 3. 几何运算逻辑 =================
        def findLaneInfo(i):
            currect_FID = records[i].record[3]
            lane_type = records[i].record[0]

            if lane_type == 'arrow':
                # 因为上面的预处理中，仅有 arrow 的 center 被完美转换了
                # 我们在这里直接读取转换好的结果，其他的一概不变，且不再重复转换
                center_new = records[i].record[2]
                return 0, 0, 0.0, center_new

            # 如果不是 arrow，提取的 center 就是原汁原味的原始字符串
            center = records[i].record[2]
            border_points = border[i].points

            dx = border_points[1][0] - border_points[0][0]
            a_1 = (border_points[1][1] - border_points[0][1]) / dx if dx != 0 else float('inf')
            currect_line_angle = math.atan2((border_points[1][1] - border_points[0][1]),
                                            (border_points[1][0] - border_points[0][0])) * 180 / 3.1415926
            currect_midpoint_x = (border_points[1][0] + border_points[0][0]) / 2
            currect_midpoint_y = (border_points[1][1] + border_points[0][1]) / 2

            if abs(a_1) < 0.01:
                current_A = Point(currect_midpoint_x, currect_midpoint_y - 20)
                current_B = Point(currect_midpoint_x, currect_midpoint_y + 20)
            else:
                A = -1 / a_1
                B = currect_midpoint_y - A * currect_midpoint_x

                vertical_line_startP_x = currect_midpoint_x + 20
                vertical_line_startP_y = vertical_line_startP_x * A + B
                vertical_line_endP_x = currect_midpoint_x - 20
                vertical_line_endP_y = vertical_line_endP_x * A + B

                vertical_line_startP = Point(vertical_line_startP_x, vertical_line_startP_y)
                vertical_line_endP = Point(vertical_line_endP_x, vertical_line_endP_y)

                current_A = get_point_by_offset(Point(currect_midpoint_x, currect_midpoint_y), vertical_line_startP, 20)
                current_B = get_point_by_offset(Point(currect_midpoint_x, currect_midpoint_y), vertical_line_endP, 20)

            candidate_line = []
            for j in range(len(border)):
                search_FID = records[j].record[3]
                if search_FID == currect_FID or records[j].record[0] != 'lane':
                    continue
                else:
                    search_points = border[j].points
                    search_line_angle = math.atan2((search_points[1][1] - search_points[0][1]),
                                                   (search_points[1][0] - search_points[0][0])) * 180 / 3.1415926

                    search_C = Point(search_points[0][0], search_points[0][1])
                    search_D = Point(search_points[1][0], search_points[1][1])
                    angle_threshold = 10
                    if records[j].record[6] != 0:
                        angle_threshold = 60

                    if abs(currect_line_angle - search_line_angle) < angle_threshold and is_intersected(current_A, current_B,
                                                                                                        search_C, search_D):
                        candidate_line.append([j, search_FID])

            lane_num = len(candidate_line)
            label = 1
            min_left_dist = float('inf')
            closest_left_line = None

            p1 = Point(border_points[0][0], border_points[0][1])
            p2 = Point(border_points[1][0], border_points[1][1])

            for cid in candidate_line:
                start_point = border[cid[0]].points[0]
                end_point = border[cid[0]].points[1]
                mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

                direction = (end_point[0] - start_point[0]) * (border_points[0][1] - mid_point[1]) - (
                        end_point[1] - start_point[1]) * (border_points[0][0] - mid_point[0])

                if direction > 0:
                    label = label + 1
                    if label > lane_num:
                        label = -1
                        return 0, label, 0.0, center
                else:
                    q1 = Point(start_point[0], start_point[1])
                    q2 = Point(end_point[0], end_point[1])
                    dist = parallel_line_distance(p1, p2, q1, q2)
                    if dist < min_left_dist:
                        min_left_dist = dist
                        closest_left_line = cid[0]

            if min_left_dist == float('inf'):
                return 0, 0, 0.0, 0

            LaneWidth = min_left_dist
            return lane_num, label, LaneWidth, center


        # ================= 4. 写入新文件 =================
        fields = file.fields[1:]
        field_names = [field[0] for field in fields]

        w = stack.enter_context(shapefile.Writer(str(outshp), shapeType=shapefile.POLYLINE))
        for field in fields:
            w.field(*field)

        for i, record in enumerate(records):
            attributes = record.record

            lane_num, label, LaneWidth, center = findLaneInfo(i)

            new_record = []
            for field_name in field_names:
                if field_name == 'LaneNumber':
                    new_record.append(lane_num)
                elif field_name == 'Label':
                    new_record.append(label)
                elif field_name == 'LaneWidth':
                    new_record.append(LaneWidth)
                elif field_name == 'center':
                    new_record.append(center)
                else:
                    new_record.append(attributes[field_names.index(field_name)])

            w.record(*new_record)
            w.shape(record.shape)

        w.close()

        write_crs(outshp, CRS.from_epsg(32650))
        print("Saved lane attributes:", outshp)
        return outshp


def main(argv=None):
    args = attribute_parser(__doc__).parse_args(argv)
    process(args.input, args.output, input_crs=args.input_crs)


if __name__ == "__main__":
    main()
