"""深度学习模型实现 - CNN、RNN、LSTM、BERT、Transformer等"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional


# ==================== 分类模型 ====================

class CNNClassifier:
    """卷积神经网络分类器"""

    def __init__(self, conv_layers: List[Tuple[int, int]] = None,
                 dense_layers: List[int] = None,
                 input_shape: Tuple[int, ...] = None,
                 num_classes: int = 2,
                 epochs: int = 50,
                 batch_size: int = 32,
                 dropout: float = 0.5,
                 verbose: int = 0):

        self.conv_layers = conv_layers or [(32, 3), (64, 3)]
        self.dense_layers = dense_layers or [128]
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.model = None
        self.history = None

    def _build_model(self):
        """构建模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            model = models.Sequential()

            # 输入层
            model.add(layers.Input(shape=self.input_shape))

            # 卷积层
            for i, (filters, kernel_size) in enumerate(self.conv_layers):
                if i == 0:
                    model.add(layers.Conv1D(filters, kernel_size, activation='relu', padding='same'))
                else:
                    model.add(layers.Conv1D(filters, kernel_size, activation='relu', padding='same'))
                model.add(layers.MaxPooling1D(2))
                model.add(layers.Dropout(self.dropout))

            # 展平
            model.add(layers.Flatten())

            # 全连接层
            for units in self.dense_layers:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(self.dropout))

            # 输出层
            if self.num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(self.num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'

            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, validation_data=None, **kwargs):
        """训练模型"""
        if self.model is None:
            # 自动推断输入形状
            if len(X.shape) == 2:
                self.input_shape = (X.shape[1], 1)
                X = X.reshape(X.shape[0], X.shape[1], 1)
            elif len(X.shape) == 3:
                self.input_shape = (X.shape[1], X.shape[2])

            self._build_model()

        # 处理标签
        if self.num_classes == 2 and len(y.shape) == 1:
            y = y

        callbacks = []
        if validation_data is not None:
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(patience=10, restore_best_weights=True))
            val_data = validation_data
        else:
            val_data = None

        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=self.verbose,
            **kwargs
        )

        return self

    def predict(self, X):
        """预测"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        pred = self.model.predict(X, verbose=0)
        if self.num_classes == 2:
            return (pred > 0.5).astype(int).flatten()
        else:
            return np.argmax(pred, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        pred = self.model.predict(X, verbose=0)
        if self.num_classes == 2:
            return np.hstack([1 - pred, pred])
        return pred


class RNNClassifier:
    """循环神经网络分类器"""

    def __init__(self, units: int = 64, layers: int = 2,
                 input_shape: Tuple[int, ...] = None,
                 num_classes: int = 2,
                 epochs: int = 50, batch_size: int = 32,
                 verbose: int = 0):

        self.units = units
        self.layers = layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def _build_model(self):
        """构建模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            model = models.Sequential()
            model.add(layers.Input(shape=self.input_shape))

            for i in range(self.layers):
                return_sequences = i < self.layers - 1
                model.add(layers.SimpleRNN(self.units, return_sequences=return_sequences))

            if self.num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(self.num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'

            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            if len(X.shape) == 2:
                self.input_shape = (X.shape[1], 1)
                X = X.reshape(X.shape[0], X.shape[1], 1)
            elif len(X.shape) == 3:
                self.input_shape = (X.shape[1], X.shape[2])
            self._build_model()

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X):
        """预测"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        pred = self.model.predict(X, verbose=0)
        if self.num_classes == 2:
            return (pred > 0.5).astype(int).flatten()
        return np.argmax(pred, axis=1)


class LSTMClassifier:
    """LSTM分类器"""

    def __init__(self, units: int = 64, layers: int = 2,
                 input_shape: Tuple[int, ...] = None,
                 num_classes: int = 2,
                 epochs: int = 50, batch_size: int = 32,
                 dropout: float = 0.2, verbose: int = 0):

        self.units = units
        self.layers = layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.model = None

    def _build_model(self):
        """构建模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            model = models.Sequential()
            model.add(layers.Input(shape=self.input_shape))

            for i in range(self.layers):
                return_sequences = i < self.layers - 1
                model.add(layers.LSTM(self.units, return_sequences=return_sequences, dropout=self.dropout))

            if self.num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(self.num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'

            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            if len(X.shape) == 2:
                self.input_shape = (X.shape[1], 1)
                X = X.reshape(X.shape[0], X.shape[1], 1)
            elif len(X.shape) == 3:
                self.input_shape = (X.shape[1], X.shape[2])
            self._build_model()

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X):
        """预测"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        pred = self.model.predict(X, verbose=0)
        if self.num_classes == 2:
            return (pred > 0.5).astype(int).flatten()
        return np.argmax(pred, axis=1)


class BertClassifier:
    """BERT分类器（占位实现，需要transformers库）"""

    def __init__(self, model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 epochs: int = 3,
                 batch_size: int = 16,
                 max_length: int = 512,
                 verbose: int = 0):

        self.model_name = model_name
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self.model = None
        self.tokenizer = None

    def _build_model(self):
        """构建BERT模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_classes
            )
        except ImportError:
            raise ImportError("transformers未安装，请运行: pip install transformers torch")

    def fit(self, X, y, **kwargs):
        """训练模型（简化实现）"""
        self._build_model()
        # 完整实现需要数据集准备和训练器配置
        return self

    def predict(self, X):
        """预测（简化实现）"""
        if self.model is None:
            self._build_model()

        # 简化实现，返回随机预测
        return np.random.randint(0, self.num_classes, size=len(X))


# ==================== 时间序列模型 ====================

class LSTMTimeSeries:
    """LSTM时间序列预测"""

    def __init__(self, units: int = 50, layers: int = 2,
                 lookback: int = 10, forecast_horizon: int = 1,
                 epochs: int = 50, batch_size: int = 32,
                 verbose: int = 0):

        self.units = units
        self.layers = layers
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler = None

    def _build_model(self, input_shape):
        """构建模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            model = models.Sequential()
            model.add(layers.Input(shape=input_shape))

            for i in range(self.layers):
                return_sequences = i < self.layers - 1
                model.add(layers.LSTM(self.units, return_sequences=return_sequences))

            model.add(layers.Dense(self.forecast_horizon))
            model.compile(optimizer='adam', loss='mse')
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2]) if len(X.shape) == 3 else (X.shape[1], 1)
            self._build_model(input_shape)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X, verbose=0).flatten()


class GRUTimeSeries:
    """GRU时间序列预测"""

    def __init__(self, units: int = 50, layers: int = 2,
                 lookback: int = 10, forecast_horizon: int = 1,
                 epochs: int = 50, batch_size: int = 32,
                 verbose: int = 0):

        self.units = units
        self.layers = layers
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def _build_model(self, input_shape):
        """构建模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            model = models.Sequential()
            model.add(layers.Input(shape=input_shape))

            for i in range(self.layers):
                return_sequences = i < self.layers - 1
                model.add(layers.GRU(self.units, return_sequences=return_sequences))

            model.add(layers.Dense(self.forecast_horizon))
            model.compile(optimizer='adam', loss='mse')
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2]) if len(X.shape) == 3 else (X.shape[1], 1)
            self._build_model(input_shape)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X, verbose=0).flatten()


class TransformerTimeSeries:
    """Transformer时间序列预测"""

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 num_layers: int = 3, lookback: int = 20,
                 forecast_horizon: int = 1,
                 epochs: int = 50, batch_size: int = 32,
                 verbose: int = 0):

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def _build_model(self, input_shape):
        """构建Transformer模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            inputs = layers.Input(shape=input_shape)

            # 位置编码
            x = layers.Dense(self.d_model)(inputs)

            # Transformer编码器
            for _ in range(self.num_layers):
                # 自注意力
                attn = layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=self.d_model)(x, x)
                x = layers.Add()([x, attn])
                x = layers.LayerNormalization()(x)

                # 前馈网络
                ff = layers.Dense(self.d_model * 4, activation='relu')(x)
                ff = layers.Dense(self.d_model)(ff)
                x = layers.Add()([x, ff])
                x = layers.LayerNormalization()(x)

            # 全局平均池化
            x = layers.GlobalAveragePooling1D()(x)

            # 输出层
            outputs = layers.Dense(self.forecast_horizon)(x)

            model = models.Model(inputs, outputs)
            model.compile(optimizer='adam', loss='mse')
            self.model = model

        except ImportError:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2]) if len(X.shape) == 3 else (X.shape[1], 1)
            self._build_model(input_shape)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X, verbose=0).flatten()