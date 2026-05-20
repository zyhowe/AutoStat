import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('cf.csv')

# 获取列名（重命名以便使用）
df.columns = ['id', 'col1', 'col2', 'col3', 'col4', 'check']
print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())

# 取前4列
cols = ['col1', 'col2', 'col3', 'col4']
X = df[cols].values

print(f"\n数据量级: min={X.min()}, max={X.max()}")

# ========== 方法1: 原始数据 SVD ==========
print("\n" + "="*50)
print("方法1: 原始数据 SVD")
print("="*50)

U, s, Vt = np.linalg.svd(X, full_matrices=False)
print(f"奇异值: {s}")
print(f"最小奇异值对应向量: {Vt[-1, :]}")

# 归一化
nonzero_mask = np.abs(Vt[-1, :]) > 1e-10
min_abs = np.min(np.abs(Vt[-1, :][nonzero_mask]))
coeffs_norm = Vt[-1, :] / min_abs
coeffs_rounded = np.round(coeffs_norm)
print(f"归一化后: {coeffs_norm}")
print(f"四舍五入: {coeffs_rounded}")

# 验证
result = np.zeros(len(df))
for i, coeff in enumerate(coeffs_rounded):
    result += coeff * df[cols[i]]
rel_error = np.abs(result) / (np.abs(df[cols[0]]) + 1)
confidence = (rel_error < 0.01).mean()
print(f"置信度: {confidence:.4f}")

# ========== 方法2: 标准化后 SVD ==========
print("\n" + "="*50)
print("方法2: 标准化后 SVD")
print("="*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
print(f"奇异值: {s}")
print(f"最小奇异值对应向量: {Vt[-1, :]}")

# 归一化
nonzero_mask = np.abs(Vt[-1, :]) > 1e-10
min_abs = np.min(np.abs(Vt[-1, :][nonzero_mask]))
coeffs_norm = Vt[-1, :] / min_abs
coeffs_rounded = np.round(coeffs_norm)
print(f"归一化后: {coeffs_norm}")
print(f"四舍五入: {coeffs_rounded}")

# 验证（需要反标准化？不，直接用原始数据验证系数）
result = np.zeros(len(df))
for i, coeff in enumerate(coeffs_rounded):
    result += coeff * df[cols[i]]
rel_error = np.abs(result) / (np.abs(df[cols[0]]) + 1)
confidence = (rel_error < 0.01).mean()
print(f"置信度: {confidence:.4f}")

# ========== 方法3: 只取 check=0 的正常数据 ==========
print("\n" + "="*50)
print("方法3: 只取正常数据（check=0）")
print("="*50)

normal_df = df[df['check'] == 0]
abnormal_df = df[df['check'] != 0]
print(f"正常数据: {len(normal_df)} 行")
print(f"异常数据: {len(abnormal_df)} 行")

X_normal = normal_df[cols].values

# 标准化
scaler2 = StandardScaler()
X_normal_scaled = scaler2.fit_transform(X_normal)

U, s, Vt = np.linalg.svd(X_normal_scaled, full_matrices=False)
print(f"奇异值: {s}")
print(f"最小奇异值对应向量: {Vt[-1, :]}")

# 归一化
nonzero_mask = np.abs(Vt[-1, :]) > 1e-10
min_abs = np.min(np.abs(Vt[-1, :][nonzero_mask]))
coeffs_norm = Vt[-1, :] / min_abs
coeffs_rounded = np.round(coeffs_norm)
print(f"归一化后: {coeffs_norm}")
print(f"四舍五入: {coeffs_rounded}")

# 在正常数据上验证
result_normal = np.zeros(len(normal_df))
for i, coeff in enumerate(coeffs_rounded):
    result_normal += coeff * normal_df[cols[i]]
rel_error_normal = np.abs(result_normal) / (np.abs(normal_df[cols[0]]) + 1)
confidence_normal = (rel_error_normal < 0.01).mean()
print(f"正常数据置信度: {confidence_normal:.4f}")

# 在异常数据上验证
if len(abnormal_df) > 0:
    result_abnormal = np.zeros(len(abnormal_df))
    for i, coeff in enumerate(coeffs_rounded):
        result_abnormal += coeff * abnormal_df[cols[i]]
    rel_error_abnormal = np.abs(result_abnormal) / (np.abs(abnormal_df[cols[0]]) + 1)
    confidence_abnormal = (rel_error_abnormal < 0.01).mean()
    print(f"异常数据置信度: {confidence_abnormal:.4f}")
    print(f"异常数据结果值（前10个）: {result_abnormal[:10]}")