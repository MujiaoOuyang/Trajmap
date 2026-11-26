"""Flatten original 3D geometry and replace center with first/last Z.

Copied from 3dTransform2d.py. The original center field is deliberately
discarded, then appended as a new field; this does not satisfy the positional
center[2] schema of cal_shp_lane_info without an explicit schema preparation
step. GDAL is required only when conversion is called. No overwrites.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

if __package__:
    from ._shapefile_io import check_output
else:
    from _shapefile_io import check_output


def default_output_path(input_shp: str) -> str:
    """
    根据输入路径自动生成输出路径。
    例如: input.shp -> input_2d.shp
    """
    root, ext = os.path.splitext(input_shp)
    return f"{root}_2d{ext}"


def _extract_first_last_z(geom: ogr.Geometry) -> Tuple[Optional[float], Optional[float]]:
    """
    递归提取几何体（Geometry）第一个点和最后一个点的 Z 值（高程）。
    支持处理单线段（LineString）和多段线（MultiLineString）等复杂几何体。
    """
    if geom is None:
        return None, None

    # 如果当前几何体包含具体的点（例如单条线段）
    if geom.GetPointCount() > 0:
        first = geom.GetPoint(0)  # 获取起点坐标 (x, y, [z])
        last = geom.GetPoint(geom.GetPointCount() - 1)  # 获取终点坐标

        # 判断坐标是否包含 Z 值（长度大于等于3表示包含 x, y, z）
        first_z = first[2] if len(first) >= 3 else None
        last_z = last[2] if len(last) >= 3 else None
        return first_z, last_z

    # 如果当前几何体是复合几何体（例如 MultiLineString，包含多个子线段）
    geom_count = geom.GetGeometryCount()
    if geom_count <= 0:
        return None, None

    first_z = None
    last_z = None
    # 遍历所有子几何体，递归提取
    for idx in range(geom_count):
        sub_first, sub_last = _extract_first_last_z(geom.GetGeometryRef(idx))

        # 只有当全局 first_z 还没赋值时，才用第一个子几何体的起点Z值赋值
        if first_z is None and sub_first is not None:
            first_z = sub_first

        # 不断更新 last_z，循环结束时它就是最后一个子几何体的终点Z值
        if sub_last is not None:
            last_z = sub_last

    return first_z, last_z


def _fmt_z(value: Optional[float]) -> str:
    """
    格式化 Z 值。去掉不必要的小数点后的 0，如果是 None 则返回空字符串。
    """
    if value is None:
        return ""
    return f"{value:g}"


def convert_3d_to_2d(input_shp: str, output_shp: str) -> None:
    """
    核心转换逻辑：读取 3D shp，提取首尾高程，压平为 2D 并保存。
    """
    if not os.path.isfile(input_shp):
        raise FileNotFoundError(input_shp)
    output_shp = str(check_output(output_shp, (input_shp,)))
    from osgeo import ogr

    # 1. 打开原始数据源
    src_ds = ogr.Open(input_shp)
    if src_ds is None:
        raise RuntimeError(f"Failed to open input shapefile: {input_shp}")

    # 获取原始图层、属性表结构和空间参考（坐标系）
    src_lyr = src_ds.GetLayer(0)
    src_defn = src_lyr.GetLayerDefn()
    src_srs = src_lyr.GetSpatialRef()

    # 2. 准备输出数据源
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 将原始的 3D 几何类型（如 LineString25D）转换为对应的 2D 类型（如 LineString）
    out_geom_type = ogr.GT_Flatten(src_lyr.GetGeomType())
    out_ds = driver.CreateDataSource(output_shp)
    if out_ds is None:
        raise RuntimeError(f"Failed to create output shapefile: {output_shp}")

    # 创建新的 2D 图层
    out_name = os.path.splitext(os.path.basename(output_shp))[0]
    out_lyr = out_ds.CreateLayer(out_name, srs=src_srs, geom_type=out_geom_type)

    # 3. 复制原始属性表字段结构，但跳过原本可能叫 "center" 的字段
    copied_field_names = []
    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        if field_defn.GetName().lower() == "center":
            continue  # 忽略旧的 center 字段
        out_lyr.CreateField(field_defn)
        copied_field_names.append(field_defn.GetName())

    # 创建新的 "center" 字段，类型为字符串，最大长度 254
    height_field = ogr.FieldDefn("center", ogr.OFTString)
    height_field.SetWidth(254)
    out_lyr.CreateField(height_field)

    out_defn = out_lyr.GetLayerDefn()

    # 4. 遍历并转换每一个几何要素（Feature）
    for src_feat in src_lyr:
        out_feat = ogr.Feature(out_defn)

        # 复制原有的属性值
        for field_name in copied_field_names:
            out_feat.SetField(field_name, src_feat.GetField(field_name))

        # 处理几何体与高程
        src_geom = src_feat.GetGeometryRef()
        if src_geom is not None:
            # 提取起点和终点的 Z 值
            start_z, end_z = _extract_first_last_z(src_geom)

            # 将提取的值格式化为 "start_z,end_z" 并写入 center 字段
            out_feat.SetField("center", f"{_fmt_z(start_z)},{_fmt_z(end_z)}")

            # 复制原始几何体，压平为 2D（丢弃所有的 Z 坐标），并赋给新要素
            out_geom = src_geom.Clone()
            out_geom.FlattenTo2D()
            out_feat.SetGeometry(out_geom)
        else:
            # 如果要素没有几何体，写入一个逗号作为空记录
            out_feat.SetField("center", ",")

        # 将构建好的新要素写入到输出图层
        out_lyr.CreateFeature(out_feat)
        out_feat = None  # 清理内存

    # 5. 安全关闭数据源，确保数据落盘
    src_ds = None
    out_ds = None


def main(argv=None) -> None:
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--input", dest="input_shp", required=True, help="Input 3D shp path")
    parser.add_argument("--output", dest="output_shp", required=True, help="New output 2D shp path; no overwrites")
    args = parser.parse_args(argv)

    input_shp = args.input_shp
    output_shp = args.output_shp

    # 检查输入文件是否存在
    if not os.path.exists(input_shp):
        raise FileNotFoundError(f"Input shp not found: {input_shp}")

    # 执行转换
    convert_3d_to_2d(input_shp, output_shp)
    print(f"Done. Output: {output_shp}")


if __name__ == "__main__":
    main()
