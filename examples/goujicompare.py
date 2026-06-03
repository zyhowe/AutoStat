import csv
import os
from collections import defaultdict


def read_csv(file_path):
    """读取CSV文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                left = row[0].strip()
                right = row[1].strip()
                data.append((left, right))
    return data


def analyze_partial_common(file_paths, output_file="partial_common_analysis.csv"):
    """
    分析部分共有的公式
    对于每个文件，列出每条公式出现在哪些文件中
    """
    file_names = [os.path.basename(f) for f in file_paths]
    all_data = {}

    # 读取所有文件
    for fname, fpath in zip(file_names, file_paths):
        all_data[fname] = read_csv(fpath)

    # 构建公式到文件列表的映射（无序对）
    formula_to_files = defaultdict(list)

    for fname, data in all_data.items():
        for left, right in data:
            formula = frozenset([left, right])
            formula_to_files[formula].append(fname)

    # 分析每个文件
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        # 全局统计
        writer.writerow(['=== 全局统计：每条公式出现在哪些文件 ==='])
        writer.writerow(['等号前', '等号后', '出现次数', '出现在以下文件'])
        for formula, files in sorted(formula_to_files.items(), key=lambda x: -len(x[1])):
            items = sorted(list(formula))
            if len(items) == 2:
                writer.writerow([
                    items[0],
                    items[1],
                    len(files),
                    '、'.join(sorted(files))
                ])
        writer.writerow([])

        # 每个文件的详细分析
        for fname in file_names:
            writer.writerow([f'=== {fname} 的公式构成分析 ==='])
            writer.writerow(['公式类型', '等号前', '等号后', '出现在以下文件', '文件数量'])

            data = all_data[fname]

            # 分类统计
            for left, right in data:
                formula = frozenset([left, right])
                files_with_formula = sorted(formula_to_files[formula])
                file_count = len(files_with_formula)

                # 确定公式类型
                if file_count == len(file_names):
                    formula_type = '所有文件共有'
                elif file_count == 1:
                    formula_type = '独有'
                else:
                    formula_type = f'部分共有（{file_count}/{len(file_names)}个文件）'

                writer.writerow([
                    formula_type,
                    left,
                    right,
                    '、'.join(files_with_formula),
                    file_count
                ])

            # 统计汇总
            writer.writerow([])
            total = len(data)
            all_common = sum(1 for left, right in data
                             if len(formula_to_files[frozenset([left, right])]) == len(file_names))
            unique = sum(1 for left, right in data
                         if len(formula_to_files[frozenset([left, right])]) == 1)
            partial = total - all_common - unique

            writer.writerow([f'=== {fname} 统计汇总 ==='])
            writer.writerow(['总公式数', total])
            writer.writerow(['所有文件共有', all_common])
            writer.writerow(['部分共有', partial])
            writer.writerow(['独有', unique])
            writer.writerow([])

    print(f"部分共有分析已保存到 {output_file}")

    # 打印摘要
    print("\n=== 各文件公式构成摘要 ===")
    for fname in file_names:
        total = len(all_data[fname])
        all_common = sum(1 for left, right in all_data[fname]
                         if len(formula_to_files[frozenset([left, right])]) == len(file_names))
        unique = sum(1 for left, right in all_data[fname]
                     if len(formula_to_files[frozenset([left, right])]) == 1)
        partial = total - all_common - unique

        print(f"\n{fname}:")
        print(f"  总数: {total}")
        print(f"  所有文件共有: {all_common} 条")
        print(f"  部分共有: {partial} 条")
        print(f"  独有: {unique} 条")

    return formula_to_files


# 使用示例
if __name__ == "__main__":
    csv_files1 = [
        "formulas_output1.csv",
        "formulas_output2.csv",
        "formulas_output3.csv",
        "formulas_output4.csv",
        "formulas_output5.csv",
        "formulas_output6.csv",
        #"formulas_output7.csv",
        #"formulas_output8.csv",
        "formulas_output9.csv"
    ]

    csv_files2 = [
        "formulas_output2.csv",
        "formulas_output9.csv"
    ]

    csv_files3 = [
        "formulas_output2.csv",
        "formulas_output8.csv"
    ]

    csv_files = [
        "formulas_output7.csv",
        "formulas_output8.csv"
    ]

    # 检查文件是否存在
    if all(os.path.exists(f) for f in csv_files):
        result = analyze_partial_common(csv_files, "partial_common_analysis4.csv")
    else:
        print("请确保所有CSV文件存在")
        print("当前目录下的CSV文件：", [f for f in os.listdir('.') if f.endswith('.csv')])