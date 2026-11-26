"""Original strict field merge and arrow extraction, with explicit paths.

No reprojection, deduplication, or topology reconstruction is performed.
Inputs require matching declared CRSs. Outputs never overwrite existing files.
Arrow ID coalescing and int32 conversion retain the original behavior.
"""
import os
from contextlib import ExitStack

if __package__:
    from ._shapefile_io import check_output, read_crs, write_crs
else:
    from _shapefile_io import check_output, read_crs, write_crs


def twofileFusion_strict(file1_path, file2_path, outshp):
    import shapefile

    outshp = check_output(outshp, (file1_path, file2_path))
    crs = read_crs(file1_path)
    if crs != read_crs(file2_path):
        raise ValueError("CRS mismatch between merge inputs; reproject explicitly first.")
    for path in (file1_path, file2_path):
        with shapefile.Reader(str(path)) as reader:
            if reader.shapeType != shapefile.POLYLINE:
                raise ValueError("Strict merge requires 2D polyline inputs; convert first.")

    with ExitStack() as stack:
        # 1. 明确我们需要保留的目标字段名 (统一使用大写的 Id)
        target_fields = ['type', 'lane_type', 'center', 'Id', 'source', 'target', 'Label', 'LaneNumber', 'LaneWidth',
                         'RAppend']

        # 创建写入器
        w = stack.enter_context(shapefile.Writer(str(outshp), shapeType=shapefile.POLYLINE))

        # 2. 为新 shp 建立极其严格的规范字段表
        # (C:字符串, N:整数, F:浮点数。如果不确定类型，全部设为字符串 'C' 也是最安全的方案)
        w.field('type', 'C', 50)
        w.field('lane_type', 'C', 50)
        w.field('center', 'C', 254)
        w.field('Id', 'N')  # 统一的 ID 字段
        w.field('source', 'C', 50)  # source 和 target 有时是长字符串，用 C 保险
        w.field('target', 'C', 50)
        w.field('Label', 'C', 50)
        w.field('LaneNumber', 'N')
        w.field('LaneWidth', 'F', 10, 4)  # 浮点数，保留4位小数
        w.field('RAppend', 'C', 50)

        # 3. 提取数据的通用处理函数
        def append_records_from_file(reader_obj):
            for record in reader_obj.shapeRecords():
                # 将这条记录的属性转为字典，方便按名称查找，比如 {'id': 123, 'type': 'solid', ...}
                record_dict = record.record.as_dict()

                # 【核心处理】：解决 id 和 Id 的大小写差异
                # 如果当前记录里有小写的 'id'，把它提取出来赋给大写的 'Id' 键
                if 'id' in record_dict:
                    record_dict['Id'] = record_dict['id']

                # 按照我们 target_fields 规定的绝对顺序，去字典里抓取值
                # 如果原本的 shp 里没有这个字段，get() 会默认返回 None，完美防止错位报错
                new_record = [record_dict.get(field_name, None) for field_name in target_fields]

                # 写入属性和几何图形
                w.record(*new_record)
                w.shape(record.shape)

        # 4. 读取并安全追加两个文件
        print("正在合并 File 1 ...")
        with shapefile.Reader(str(file1_path)) as file1:
            append_records_from_file(file1)

        print("正在合并 File 2 ...")
        with shapefile.Reader(str(file2_path)) as file2:
            append_records_from_file(file2)

        w.close()

        write_crs(outshp, crs)
        return outshp


def extract_and_merge_arrows(base_shp_path, multi_shp_folder, output_shp_path):
    import geopandas as gpd
    import pandas as pd

    output_shp_path = check_output(output_shp_path, (base_shp_path,))
    base_crs = read_crs(base_shp_path)
    if not os.path.isdir(multi_shp_folder):
        raise NotADirectoryError(multi_shp_folder)

    # ================= 2. 读取基础底图 =================
    print(f"正在读取底图: {base_shp_path} ...")
    base_gdf = gpd.read_file(base_shp_path)
    arrow_features_list = []

    # ================= 3. 遍历文件夹，提取 Arrow =================
    print(f"\n开始扫描文件夹: {multi_shp_folder}")
    for filename in sorted(os.listdir(multi_shp_folder)):
        if filename.lower().endswith('.shp'):
            filepath = os.path.join(multi_shp_folder, filename)
            if read_crs(filepath) != base_crs:
                raise ValueError("CRS mismatch in arrow input: " + filepath)
            temp_gdf = gpd.read_file(filepath)

            if 'type' in temp_gdf.columns:
                # 提取 type 为 arrow 的要素
                arrow_mask = temp_gdf['type'].astype(str).str.lower() == 'arrow'
                arrow_subset = temp_gdf[arrow_mask].copy()  # 加上 .copy() 防止 pandas 报链式赋值警告

                if not arrow_subset.empty:
                    # 强迫症福音：强制将提取出来的 type 统一刷成小写的 'arrow'，防止原来写的是 'Arrow'
                    arrow_subset['type'] = 'arrow'

                    print(f"  [+] 在 {filename} 中成功提取 {len(arrow_subset)} 个 arrow 要素！")
                    arrow_features_list.append(arrow_subset)

    # ================= 4. 执行数据合并 =================
    if not arrow_features_list:
        print("\n扫描结束，没有找到任何 arrow 要素。")
        return

    print(f"\n扫描结束，准备将 {len(arrow_features_list)} 批 arrow 数据合并到底图中...")
    combined_gdf = pd.concat([base_gdf] + arrow_features_list, ignore_index=True)

    # ================= 5. 【核心修复】彻底解决大小写导致的多重列名问题 =================
    id_cols = [c for c in combined_gdf.columns if c.lower() == 'id']
    if id_cols:
        print(f"检测到以下 ID 列变体 {id_cols}，正在融合...")
        unified_id = combined_gdf[id_cols[0]].copy()
        for c in id_cols[1:]:
            unified_id = unified_id.fillna(combined_gdf[c])
        combined_gdf = combined_gdf.drop(columns=id_cols)
        combined_gdf['id'] = unified_id

    # ================= 6. 【核心修复】解决 ArcMap 双精度问题 =================
    if 'id' in combined_gdf.columns:
        print("正在强制将 id 转换为 ArcMap 长整型 (int32)...")
        combined_gdf['id'] = combined_gdf['id'].fillna(0).astype('int32')

    # ================= 7. 严格规范最终输出字段 =================
    target_cols = ['type', 'lane_type', 'center', 'id', 'source', 'target',
                   'Label', 'LaneNumber', 'LaneWidth', 'RAppend', 'geometry']

    # 这里需要非常小心地去重，防止原来数据里连 'type' 都有大写和小写导致重复
    final_cols = []
    for c in target_cols:
        if c in combined_gdf.columns and c not in final_cols:
            final_cols.append(c)

    # 只提取最终的干净列（如果在上面步骤剔除了重复列，这里切片就绝对安全）
    # 为了防止因为旧 pandas 版本带来的切片重复问题，加一个 .loc[:,~combined_gdf.columns.duplicated()] 终极保险
    combined_gdf = combined_gdf.loc[:, ~combined_gdf.columns.duplicated()]
    final_gdf = combined_gdf[final_cols]

    # ================= 8. 保存为 Shapefile =================
    print(f"正在保存最终 Shapefile: {output_shp_path} ...")
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs=base_crs)
    final_gdf.to_file(output_shp_path, encoding='utf-8', index=False)

    print("Arrow merge saved:", output_shp_path)
    return output_shp_path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge", help="Strict two-file field merge.")
    merge.add_argument("--input1", required=True)
    merge.add_argument("--input2", required=True)
    merge.add_argument("--output", required=True)
    arrows = commands.add_parser("arrows", help="Append type=arrow features from a folder.")
    arrows.add_argument("--base", required=True)
    arrows.add_argument("--arrow-dir", required=True)
    arrows.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        twofileFusion_strict(args.input1, args.input2, args.output)
    else:
        extract_and_merge_arrows(args.base, args.arrow_dir, args.output)


if __name__ == "__main__":
    main()
