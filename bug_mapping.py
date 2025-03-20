import csv
import json
import os

# 配置路径
csv_file_path = "your_file.csv"  # 你的 CSV 文件路径
output_json_path = "bug_mappings.json"  # 输出 JSON 文件

# 结果存储
bug_mappings = {}

# 读取 CSV 并构建映射
with open(csv_file_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # 跳过表头

    for idx, row in enumerate(reader, start=1):
        bug_report_name = f"bug_report_{idx}.json"
        bug_files = row[4]  # 第五列是 bug file

        # 处理路径中的 `/` 变为 `_`
        processed_files = [file.replace("/", "_") for file in bug_files.split(",")]

        # 存储对应关系
        bug_mappings[bug_report_name] = processed_files

# 保存为 JSON
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(bug_mappings, json_file, indent=2)

print(f"✅ 处理完成，结果已保存到 {output_json_path}")
