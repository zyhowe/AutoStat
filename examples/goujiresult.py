import re
import csv

# 原始文本数据
text = """
  [DEBUG]     1. companyfixasset33 = companyfixasset51 (置信度=1.0, 有效行数=14, 满足行数=14)
  [DEBUG]     2. companyfixasset37 = companyfixasset38 (置信度=0.8656, 有效行数=253, 满足行数=219)
  [DEBUG]     3. companyfixasset43 = companyfixasset44 (置信度=0.8701, 有效行数=254, 满足行数=221)
  [DEBUG]     4. companyfixasset4 = companyfixasset50 + companyfixasset18 (置信度=0.7866, 有效行数=4311, 满足行数=3391)
  [DEBUG]     5. companyfixasset11 = companyfixasset16 + companyfixasset12 (置信度=0.7482, 有效行数=691, 满足行数=517)
  [DEBUG]     6. companyfixasset17 = companyfixasset53 + companyfixasset31 (置信度=0.7862, 有效行数=4355, 满足行数=3424)
  [DEBUG]     7. companyfixasset4 + companyfixasset53 + companyfixasset31 = companyfixasset50 + companyfixasset18 + companyfixasset17 (置信度=0.8659, 有效行数=4206, 满足行数=3642)
  [DEBUG]     8. companyfixasset25 + companyfixasset11 = companyfixasset12 + companyfixasset26 (置信度=0.5765, 有效行数=2071, 满足行数=1194)
  [DEBUG]     9. companyfixasset25 = companyfixasset28 + companyfixasset26 (置信度=0.7899, 有效行数=119, 满足行数=94)
  [DEBUG]     10. companyfixasset11 = companyfixasset14 + companyfixasset12 (置信度=0.6774, 有效行数=124, 满足行数=84)
  [DEBUG]     11. companyfixasset18 + companyfixasset14 + companyfixasset50 + companyfixasset17 + companyfixasset12 = companyfixasset4 + companyfixasset11 + companyfixasset53 + companyfixasset31 (置信度=0.6048, 有效行数=124, 满足行数=75)
  [DEBUG]     12. companyfixasset28 + companyfixasset14 + companyfixasset12 + companyfixasset26 = companyfixasset25 + companyfixasset11 (置信度=0.6637, 有效行数=113, 满足行数=75)
  [DEBUG]     13. companyfixasset5 = companyfixasset8 + companyfixasset6 (置信度=0.5337, 有效行数=1171, 满足行数=625)
  [DEBUG]     14. companyfixasset38 + companyfixasset19 = companyfixasset20 + companyfixasset37 (置信度=0.576, 有效行数=250, 满足行数=144)
  [DEBUG]     15. companyfixasset38 + companyfixasset43 = companyfixasset37 + companyfixasset44 (置信度=0.7244, 有效行数=127, 满足行数=92)
  [DEBUG]     16. pagenum + companyfixasset19 = companyfixasset20 (置信度=0.5474, 有效行数=2205, 满足行数=1207)
  [DEBUG]     17. companyfixasset19 = companyfixasset24 + companyfixasset20 (置信度=0.833, 有效行数=443, 满足行数=369)
  [DEBUG]     18. companyfixasset26 + companyfixasset30 = companyfixasset25 (置信度=0.7138, 有效行数=615, 满足行数=439)
  [DEBUG]     19. companyfixasset26 + companyfixasset11 + companyfixasset30 = companyfixasset25 + companyfixasset16 + companyfixasset12 (置信度=0.7375, 有效行数=560, 满足行数=413)
  [DEBUG]     20. companyfixasset6 + companyfixasset10 + companyfixasset8 = companyfixasset5 (置信度=0.6508, 有效行数=358, 满足行数=233)
  [DEBUG]     21. companyfixasset20 + companyfixasset43 = companyfixasset44 + companyfixasset19 (置信度=0.5472, 有效行数=254, 满足行数=139)
  [DEBUG]     22. companyfixasset18 + companyfixasset19 = companyfixasset25 + companyfixasset31 (置信度=0.9727, 有效行数=3228, 满足行数=3140)
  [DEBUG]     23. companyfixasset4 = companyfixasset18 + companyfixasset32 (置信度=0.9847, 有效行数=852, 满足行数=839)
  [DEBUG]     24. companyfixasset53 + companyfixasset32 = companyfixasset50 + companyfixasset35 (置信度=0.8106, 有效行数=808, 满足行数=655)
  [DEBUG]     25. companyfixasset32 + companyfixasset25 + companyfixasset5 = companyfixasset11 + companyfixasset35 + companyfixasset19 (置信度=0.986, 有效行数=643, 满足行数=634)
  [DEBUG]     26. companyfixasset11 = companyfixasset13 + companyfixasset12 (置信度=0.5964, 有效行数=223, 满足行数=133)
  [DEBUG]     27. companyfixasset25 = companyfixasset27 + companyfixasset26 (置信度=0.6028, 有效行数=214, 满足行数=129)
  [DEBUG]     28. companyfixasset21 + companyfixasset20 = companyfixasset19 (置信度=0.8068, 有效行数=383, 满足行数=309)
  [DEBUG]     29. companyfixasset4 + companyfixasset19 = companyfixasset32 + companyfixasset31 + companyfixasset25 (置信度=0.997, 有效行数=673, 满足行数=671)
  [DEBUG]     30. companyfixasset35 = companyfixasset49 + companyfixasset53 (置信度=0.9867, 有效行数=300, 满足行数=296)
  [DEBUG]     31. companyfixasset6 + companyfixasset20 + companyfixasset8 = companyfixasset5 + companyfixasset19 (置信度=0.5195, 有效行数=1153, 满足行数=599)
  [DEBUG]     32. companyfixasset25 + companyfixasset24 + companyfixasset20 = companyfixasset30 + companyfixasset19 + companyfixasset26 (置信度=0.6368, 有效行数=201, 满足行数=128)
  [DEBUG]     33. companyfixasset5 + companyfixasset4 = companyfixasset16 + companyfixasset12 + companyfixasset31 + companyfixasset53 (置信度=0.5095, 有效行数=632, 满足行数=322)
  [DEBUG]     34. companyfixasset25 + companyfixasset49 + companyfixasset31 + companyfixasset53 = companyfixasset19 + companyfixasset18 + companyfixasset35 (置信度=0.9825, 有效行数=285, 满足行数=280)
  [DEBUG]     35. companyfixasset32 = companyfixasset36 + companyfixasset50 (置信度=0.993, 有效行数=285, 满足行数=283)
  [DEBUG]     36. companyfixasset25 + companyfixasset36 + companyfixasset5 + companyfixasset50 = companyfixasset19 + companyfixasset35 + companyfixasset11 (置信度=0.9892, 有效行数=278, 满足行数=275)
  [DEBUG]     37. companyfixasset17 = companyfixasset35 + companyfixasset31 (置信度=0.9781, 有效行数=867, 满足行数=848)
  [DEBUG]     38. companyfixasset36 + companyfixasset37 = companyfixasset43 + companyfixasset49 (置信度=0.9826, 有效行数=287, 满足行数=282)
  [DEBUG]     39. companyfixasset43 + companyfixasset49 = pagenum + companyfixasset36 + companyfixasset37 (置信度=0.8246, 有效行数=211, 满足行数=174)
  [DEBUG]     40. companyfixasset43 + companyfixasset25 + companyfixasset5 + companyfixasset50 = companyfixasset19 + companyfixasset37 + companyfixasset11 + companyfixasset53 (置信度=0.9817, 有效行数=273, 满足行数=268)
  [DEBUG]     41. companyfixasset5 = companyfixasset51 + companyfixasset19 (置信度=0.7368, 有效行数=38, 满足行数=28)
  [DEBUG]     42. companyfixasset11 = companyfixasset52 + companyfixasset25 (置信度=0.6129, 有效行数=31, 满足行数=19)
  [DEBUG]     43. companyfixasset4 + companyfixasset25 + companyfixasset5 + companyfixasset31 = companyfixasset19 + companyfixasset18 + companyfixasset17 + companyfixasset11 (置信度=0.9708, 有效行数=3014, 满足行数=2926)
  [DEBUG]     44. companyfixasset50 + companyfixasset25 + companyfixasset5 = companyfixasset19 + companyfixasset53 + companyfixasset11 (置信度=0.7848, 有效行数=2909, 满足行数=2283)
  [DEBUG]     45. companyfixasset42 + companyfixasset38 = companyfixasset37 (置信度=1.0, 有效行数=21, 满足行数=21)
  [DEBUG]     46. companyfixasset23 + companyfixasset20 = companyfixasset19 (置信度=0.6733, 有效行数=150, 满足行数=101)
  [DEBUG]     47. companyfixasset51 + companyfixasset52 = companyfixasset33 + companyfixasset34 (置信度=1.0, 有效行数=6, 满足行数=6)
  [DEBUG]     48. companyfixasset50 + companyfixasset52 = companyfixasset32 + companyfixasset34 (置信度=1.0, 有效行数=10, 满足行数=10)
  [DEBUG]     49. companyfixasset33 + companyfixasset32 = companyfixasset51 + companyfixasset50 (置信度=1.0, 有效行数=12, 满足行数=12)
  [DEBUG]     50. companyfixasset5 + companyfixasset19 = companyfixasset21 + companyfixasset20 + companyfixasset8 + companyfixasset7 + companyfixasset6 (置信度=0.5587, 有效行数=179, 满足行数=100)
  [DEBUG]     51. companyfixasset17 + companyfixasset11 = companyfixasset4 + companyfixasset5 (置信度=0.9709, 有效行数=3165, 满足行数=3073)
  [DEBUG]     52. companyfixasset34 + companyfixasset35 = companyfixasset33 + companyfixasset32 (置信度=1.0, 有效行数=23, 满足行数=23)
  [DEBUG]     53. companyfixasset25 + companyfixasset34 = companyfixasset11 (置信度=0.6875, 有效行数=32, 满足行数=22)
  [DEBUG]     54. companyfixasset34 + companyfixasset50 + companyfixasset25 = companyfixasset32 + companyfixasset11 (置信度=0.5, 有效行数=24, 满足行数=12)
  [DEBUG]     55. companyfixasset5 = companyfixasset33 + companyfixasset19 (置信度=0.931, 有效行数=29, 满足行数=27)
  [DEBUG]     56. companyfixasset19 + companyfixasset50 + companyfixasset18 = companyfixasset32 + companyfixasset31 + companyfixasset25 (置信度=0.6132, 有效行数=636, 满足行数=390)
  [DEBUG]     57. companyfixasset32 + companyfixasset30 = companyfixasset19 + companyfixasset16 (置信度=0.5714, 有效行数=7, 满足行数=4)

  """


def extract_formulas(text):
    """从文本中提取公式并处理"""
    results = []

    for line in text.strip().split('\n'):
        # 匹配等号前后的表达式
        match = re.search(r'=\s*(.+?)\s*\(置信度', line)
        if not match:
            continue

        # 提取等号右边
        right_expr = match.group(1).strip()

        # 提取等号左边（等号之前的部分，去掉序号）
        left_part = re.search(r'\]\s*(?:\d+\.\s+)?(.+?)\s*=', line)
        if left_part:
            left_expr = left_part.group(1).strip()
        else:
            continue

        # 处理等号左边：分割、排序、用顿号连接
        left_items = [item.strip() for item in left_expr.split('+')]
        left_sorted = sorted(left_items)
        left_result = '、'.join(left_sorted)

        # 处理等号右边：分割、排序、用顿号连接
        right_items = [item.strip() for item in right_expr.split('+')]
        right_sorted = sorted(right_items)
        right_result = '、'.join(right_sorted)

        results.append((left_result, right_result))

    return results


def save_to_csv(results, csv_file="formulas_output.csv"):
    """保存结果到CSV文件"""
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['等号前', '等号后'])
        writer.writerows(results)
    print(f"结果已保存到 {csv_file}")


def print_results(results):
    """打印结果（完整显示，不省略）"""
    print(f"{'序号':<4} {'等号前':<60} {'等号后'}")
    print("-" * 120)
    for i, (left, right) in enumerate(results, 1):
        print(f"{i:<4} {left:<60} {right}")


# 主程序
if __name__ == "__main__":
    # 提取公式
    formulas = extract_formulas(text)

    # 打印结果
    print_results(formulas)

    # 保存到CSV文件
    save_to_csv(formulas)

    # 输出统计信息
    print(f"\n共提取 {len(formulas)} 条公式")