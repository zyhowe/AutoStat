"""数据预处理模块 - 标准化数据预处理"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os


class DataPreprocessor:
    """数据预处理器 - 支持多种预处理策略"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预处理器

        参数:
        - config: 预处理配置
            {
                "missing_strategy": "drop|fill_mean|fill_median|fill_mode|fill_constant",
                "missing_value": None,  # fill_constant时使用
                "scaling": "standard|minmax|robust|none",
                "encoding": "onehot|label|target|none",
                "categorical_features": [],
                "numerical_features": [],
                "target_column": None
            }
        """
        self.config = config or {}
        self.scaler = None
        self.encoders = {}
        self.imputers = {}
        self.label_encoder = None
        self.fitted = False
        self.feature_names = None
        self.categorical_features = self.config.get("categorical_features", [])
        self.numerical_features = self.config.get("numerical_features", [])
        self.target_column = self.config.get("target_column")

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None):
        """拟合预处理器"""

        # 自动识别特征类型（如果未指定）
        if not self.categorical_features and not self.numerical_features:
            self._auto_detect_features(df)

        # 处理缺失值
        self._fit_missing_handling(df)

        # 处理数值特征缩放
        self._fit_scaling(df)

        # 处理分类特征编码
        self._fit_encoding(df)

        # 保存特征名
        self.feature_names = self.categorical_features + self.numerical_features
        self.fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """转换数据"""
        if not self.fitted:
            raise ValueError("预处理器未拟合，请先调用 fit()")

        # 复制数据
        data = df.copy()

        # 处理缺失值
        data = self._transform_missing_handling(data)

        # 处理分类特征编码
        encoded_features = self._transform_encoding(data)

        # 处理数值特征缩放
        scaled_features = self._transform_scaling(data)

        # 合并特征
        all_features = []
        if scaled_features is not None:
            all_features.append(scaled_features)
        if encoded_features is not None:
            all_features.append(encoded_features)

        if not all_features:
            return np.array([])

        return np.hstack(all_features)

    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> np.ndarray:
        """拟合并转换数据"""
        self.fit(df, target)
        return self.transform(df)

    def _auto_detect_features(self, df: pd.DataFrame):
        """自动识别特征类型"""
        for col in df.columns:
            if col == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)

    def _fit_missing_handling(self, df: pd.DataFrame):
        """拟合缺失值处理器"""
        strategy = self.config.get("missing_strategy", "drop")

        if strategy == "drop":
            self.imputers = None
            return

        for col in self.numerical_features:
            if strategy == "fill_mean":
                imputer = SimpleImputer(strategy="mean")
            elif strategy == "fill_median":
                imputer = SimpleImputer(strategy="median")
            elif strategy == "fill_constant":
                imputer = SimpleImputer(strategy="constant", fill_value=self.config.get("missing_value", 0))
            else:
                continue

            imputer.fit(df[[col]])
            self.imputers[col] = imputer

        for col in self.categorical_features:
            if strategy == "fill_mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    self.imputers[col] = mode_val[0]
                else:
                    self.imputers[col] = "unknown"
            elif strategy == "fill_constant":
                self.imputers[col] = self.config.get("missing_value", "unknown")

    def _transform_missing_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换缺失值处理"""
        strategy = self.config.get("missing_strategy", "drop")

        if strategy == "drop":
            return df.dropna()

        data = df.copy()

        for col in self.numerical_features:
            if col in self.imputers:
                data[col] = self.imputers[col].transform(data[[col]])

        for col in self.categorical_features:
            if col in self.imputers:
                data[col] = data[col].fillna(self.imputers[col])

        return data

    def _fit_scaling(self, df: pd.DataFrame):
        """拟合缩放器"""
        scaling = self.config.get("scaling", "standard")

        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            return

        if self.numerical_features:
            self.scaler.fit(df[self.numerical_features])

    def _transform_scaling(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """转换缩放"""
        if not self.numerical_features:
            return None

        if self.scaler is None:
            return df[self.numerical_features].values

        return self.scaler.transform(df[self.numerical_features])

    def _fit_encoding(self, df: pd.DataFrame):
        """拟合编码器"""
        encoding = self.config.get("encoding", "onehot")

        if encoding == "label":
            for col in self.categorical_features:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = encoder
        elif encoding == "onehot":
            for col in self.categorical_features:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(df[[col]].astype(str))
                self.encoders[col] = encoder
        else:
            self.encoders = {}

    def _transform_encoding(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """转换编码"""
        if not self.categorical_features:
            return None

        encoding = self.config.get("encoding", "onehot")
        encoded_list = []

        for col in self.categorical_features:
            if col not in self.encoders:
                continue

            data = df[col].fillna("unknown").astype(str)

            if encoding == "label":
                encoded = self.encoders[col].transform(data)
                encoded_list.append(encoded.reshape(-1, 1))
            elif encoding == "onehot":
                encoded = self.encoders[col].transform(data.values.reshape(-1, 1))
                encoded_list.append(encoded)

        if not encoded_list:
            return None

        return np.hstack(encoded_list)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """逆转换目标变量（用于分类标签还原）"""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y.astype(int))
        return y

    def save(self, path: str):
        """保存预处理器"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """加载预处理器"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_feature_names(self) -> List[str]:
        """获取特征名列表"""
        if self.feature_names:
            return self.feature_names
        return []