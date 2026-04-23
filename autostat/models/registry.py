"""模型注册表 - 所有模型的元信息定义"""

MODEL_REGISTRY = {
    # ==================== 分类模型 ====================
    "classification": {
        "logistic_regression": {
            "name": "逻辑回归",
            "class": "LogisticRegression",
            "module": "sklearn.linear_model",
            "description": "线性分类器，适用于二分类问题，可输出概率",
            "params": {
                "C": {"type": "float", "default": 1.0, "range": [0.001, 100], "description": "正则化强度"},
                "max_iter": {"type": "int", "default": 1000, "range": [100, 5000], "description": "最大迭代次数"},
                "solver": {"type": "choice", "default": "lbfgs", "options": ["lbfgs", "liblinear", "saga"],
                           "description": "优化算法"},
                "penalty": {"type": "choice", "default": "l2", "options": ["l1", "l2", "elasticnet"],
                            "description": "正则化类型"}
            },
            "requirements": ["scikit-learn"]
        },
        "decision_tree": {
            "name": "决策树",
            "class": "DecisionTreeClassifier",
            "module": "sklearn.tree",
            "description": "树形结构，可解释性强，支持可视化",
            "params": {
                "max_depth": {"type": "int", "default": None, "range": [1, 50], "description": "最大深度"},
                "min_samples_split": {"type": "int", "default": 2, "range": [2, 20],
                                      "description": "节点分裂最小样本数"},
                "criterion": {"type": "choice", "default": "gini", "options": ["gini", "entropy"],
                              "description": "分裂标准"}
            },
            "requirements": ["scikit-learn"]
        },
        "random_forest": {
            "name": "随机森林",
            "class": "RandomForestClassifier",
            "module": "sklearn.ensemble",
            "description": "集成学习，抗过拟合，可输出特征重要性",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "树的数量"},
                "max_depth": {"type": "int", "default": None, "range": [1, 50], "description": "最大深度"},
                "min_samples_split": {"type": "int", "default": 2, "range": [2, 20],
                                      "description": "节点分裂最小样本数"},
                "max_features": {"type": "choice", "default": "sqrt", "options": ["sqrt", "log2", None],
                                 "description": "最大特征数"}
            },
            "requirements": ["scikit-learn"]
        },
        "xgboost": {
            "name": "XGBoost",
            "class": "XGBClassifier",
            "module": "xgboost",
            "description": "梯度提升，高精度，支持GPU加速",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "max_depth": {"type": "int", "default": 6, "range": [1, 15], "description": "树的最大深度"},
                "subsample": {"type": "float", "default": 1.0, "range": [0.5, 1.0], "description": "样本采样比例"},
                "colsample_bytree": {"type": "float", "default": 1.0, "range": [0.5, 1.0],
                                     "description": "特征采样比例"}
            },
            "requirements": ["xgboost"]
        },
        "lightgbm": {
            "name": "LightGBM",
            "class": "LGBMClassifier",
            "module": "lightgbm",
            "description": "基于直方图，速度快，内存占用小",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "num_leaves": {"type": "int", "default": 31, "range": [2, 255], "description": "叶子节点数"},
                "max_depth": {"type": "int", "default": -1, "range": [-1, 50], "description": "最大深度"}
            },
            "requirements": ["lightgbm"]
        },
        "catboost": {
            "name": "CatBoost",
            "class": "CatBoostClassifier",
            "module": "catboost",
            "description": "自动处理分类特征，无需独热编码",
            "params": {
                "iterations": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "depth": {"type": "int", "default": 6, "range": [1, 16], "description": "树深度"}
            },
            "requirements": ["catboost"]
        },
        "svm": {
            "name": "支持向量机",
            "class": "SVC",
            "module": "sklearn.svm",
            "description": "适用于小数据集、高维特征",
            "params": {
                "C": {"type": "float", "default": 1.0, "range": [0.1, 100], "description": "惩罚系数"},
                "kernel": {"type": "choice", "default": "rbf", "options": ["linear", "poly", "rbf", "sigmoid"],
                           "description": "核函数"},
                "gamma": {"type": "choice", "default": "scale", "options": ["scale", "auto"],
                          "description": "核函数系数"}
            },
            "requirements": ["scikit-learn"]
        },
        "knn": {
            "name": "K近邻",
            "class": "KNeighborsClassifier",
            "module": "sklearn.neighbors",
            "description": "无训练过程，基于距离分类",
            "params": {
                "n_neighbors": {"type": "int", "default": 5, "range": [1, 50], "description": "邻居数量"},
                "weights": {"type": "choice", "default": "uniform", "options": ["uniform", "distance"],
                            "description": "权重方式"},
                "algorithm": {"type": "choice", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"],
                              "description": "搜索算法"}
            },
            "requirements": ["scikit-learn"]
        },
        "naive_bayes": {
            "name": "朴素贝叶斯",
            "class": "GaussianNB",
            "module": "sklearn.naive_bayes",
            "description": "基于贝叶斯定理，适合文本分类",
            "params": {
                "var_smoothing": {"type": "float", "default": 1e-9, "range": [1e-12, 1e-5], "description": "方差平滑"}
            },
            "requirements": ["scikit-learn"]
        },
        "adaboost": {
            "name": "AdaBoost",
            "class": "AdaBoostClassifier",
            "module": "sklearn.ensemble",
            "description": "自适应提升，串行训练",
            "params": {
                "n_estimators": {"type": "int", "default": 50, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 1.0, "range": [0.01, 2.0], "description": "学习率"}
            },
            "requirements": ["scikit-learn"]
        },
        "gradient_boosting": {
            "name": "梯度提升",
            "class": "GradientBoostingClassifier",
            "module": "sklearn.ensemble",
            "description": "梯度提升决策树，稳定可靠",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "max_depth": {"type": "int", "default": 3, "range": [1, 10], "description": "最大深度"}
            },
            "requirements": ["scikit-learn"]
        },
        "mlp_classifier": {
            "name": "多层感知机",
            "class": "MLPClassifier",
            "module": "sklearn.neural_network",
            "description": "基础神经网络，适合中小数据集",
            "params": {
                "hidden_layer_sizes": {"type": "list", "default": [100, 50], "description": "隐藏层神经元数"},
                "activation": {"type": "choice", "default": "relu", "options": ["relu", "tanh", "logistic"],
                               "description": "激活函数"},
                "max_iter": {"type": "int", "default": 500, "range": [100, 2000], "description": "最大迭代次数"},
                "learning_rate_init": {"type": "float", "default": 0.001, "range": [0.0001, 0.1],
                                       "description": "初始学习率"}
            },
            "requirements": ["scikit-learn"]
        },
        "cnn_classifier": {
            "name": "卷积神经网络",
            "class": "CNNClassifier",
            "module": "autostat.models.deep_learning",
            "description": "深度学习，适合图像、序列数据",
            "params": {
                "conv_layers": {"type": "list", "default": [[32, 3], [64, 3]], "description": "卷积层配置"},
                "dense_layers": {"type": "list", "default": [128], "description": "全连接层配置"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256], "description": "批次大小"},
                "dropout": {"type": "float", "default": 0.5, "range": [0.0, 0.8], "description": "Dropout比例"}
            },
            "requirements": ["tensorflow"]
        },
        "rnn_classifier": {
            "name": "循环神经网络",
            "class": "RNNClassifier",
            "module": "autostat.models.deep_learning",
            "description": "适合序列数据、时间序列",
            "params": {
                "units": {"type": "int", "default": 64, "range": [16, 256], "description": "RNN单元数"},
                "layers": {"type": "int", "default": 2, "range": [1, 4], "description": "RNN层数"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256], "description": "批次大小"}
            },
            "requirements": ["tensorflow"]
        },
        "lstm_classifier": {
            "name": "LSTM",
            "class": "LSTMClassifier",
            "module": "autostat.models.deep_learning",
            "description": "长短期记忆网络，适合长序列",
            "params": {
                "units": {"type": "int", "default": 64, "range": [16, 256], "description": "LSTM单元数"},
                "layers": {"type": "int", "default": 2, "range": [1, 4], "description": "LSTM层数"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256], "description": "批次大小"}
            },
            "requirements": ["tensorflow"]
        },
        "bert_classifier": {
            "name": "BERT",
            "class": "BertClassifier",
            "module": "transformers",
            "description": "预训练语言模型，适合文本分类",
            "params": {
                "model_name": {"type": "choice", "default": "bert-base-uncased",
                               "options": ["bert-base-uncased", "bert-base-chinese", "roberta-base"],
                               "description": "预训练模型"},
                "epochs": {"type": "int", "default": 3, "range": [1, 10], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 16, "range": [4, 64], "description": "批次大小"},
                "max_length": {"type": "int", "default": 512, "range": [64, 512], "description": "最大序列长度"}
            },
            "requirements": ["transformers", "torch"]
        }
    },

    # ==================== 回归模型 ====================
    "regression": {
        "linear_regression": {
            "name": "线性回归",
            "class": "LinearRegression",
            "module": "sklearn.linear_model",
            "description": "线性关系建模，简单快速",
            "params": {
                "fit_intercept": {"type": "bool", "default": True, "description": "是否拟合截距"}
            },
            "requirements": ["scikit-learn"]
        },
        "ridge": {
            "name": "岭回归",
            "class": "Ridge",
            "module": "sklearn.linear_model",
            "description": "L2正则化，缓解多重共线性",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "range": [0.001, 100], "description": "正则化强度"}
            },
            "requirements": ["scikit-learn"]
        },
        "lasso": {
            "name": "Lasso回归",
            "class": "Lasso",
            "module": "sklearn.linear_model",
            "description": "L1正则化，可进行特征选择",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "range": [0.001, 100], "description": "正则化强度"}
            },
            "requirements": ["scikit-learn"]
        },
        "elastic_net": {
            "name": "弹性网络",
            "class": "ElasticNet",
            "module": "sklearn.linear_model",
            "description": "L1+L2混合正则化",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "range": [0.001, 100], "description": "正则化强度"},
                "l1_ratio": {"type": "float", "default": 0.5, "range": [0, 1], "description": "L1比例"}
            },
            "requirements": ["scikit-learn"]
        },
        "decision_tree_regressor": {
            "name": "决策树回归",
            "class": "DecisionTreeRegressor",
            "module": "sklearn.tree",
            "description": "树形回归，可解释性强",
            "params": {
                "max_depth": {"type": "int", "default": None, "range": [1, 50], "description": "最大深度"},
                "min_samples_split": {"type": "int", "default": 2, "range": [2, 20],
                                      "description": "节点分裂最小样本数"}
            },
            "requirements": ["scikit-learn"]
        },
        "random_forest_regressor": {
            "name": "随机森林回归",
            "class": "RandomForestRegressor",
            "module": "sklearn.ensemble",
            "description": "集成回归，抗过拟合",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "树的数量"},
                "max_depth": {"type": "int", "default": None, "range": [1, 50], "description": "最大深度"}
            },
            "requirements": ["scikit-learn"]
        },
        "xgboost_regressor": {
            "name": "XGBoost回归",
            "class": "XGBRegressor",
            "module": "xgboost",
            "description": "梯度提升回归，高精度",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "max_depth": {"type": "int", "default": 6, "range": [1, 15], "description": "最大深度"}
            },
            "requirements": ["xgboost"]
        },
        "lightgbm_regressor": {
            "name": "LightGBM回归",
            "class": "LGBMRegressor",
            "module": "lightgbm",
            "description": "轻量级梯度提升回归",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "num_leaves": {"type": "int", "default": 31, "range": [2, 255], "description": "叶子节点数"}
            },
            "requirements": ["lightgbm"]
        },
        "svr": {
            "name": "支持向量回归",
            "class": "SVR",
            "module": "sklearn.svm",
            "description": "支持向量机回归",
            "params": {
                "C": {"type": "float", "default": 1.0, "range": [0.1, 100], "description": "惩罚系数"},
                "kernel": {"type": "choice", "default": "rbf", "options": ["linear", "poly", "rbf"],
                           "description": "核函数"},
                "epsilon": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "不敏感损失"}
            },
            "requirements": ["scikit-learn"]
        },
        "knn_regressor": {
            "name": "K近邻回归",
            "class": "KNeighborsRegressor",
            "module": "sklearn.neighbors",
            "description": "基于邻居的回归",
            "params": {
                "n_neighbors": {"type": "int", "default": 5, "range": [1, 50], "description": "邻居数量"},
                "weights": {"type": "choice", "default": "uniform", "options": ["uniform", "distance"],
                            "description": "权重方式"}
            },
            "requirements": ["scikit-learn"]
        },
        "gradient_boosting_regressor": {
            "name": "梯度提升回归",
            "class": "GradientBoostingRegressor",
            "module": "sklearn.ensemble",
            "description": "梯度提升决策树回归",
            "params": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 500], "description": "迭代次数"},
                "learning_rate": {"type": "float", "default": 0.1, "range": [0.01, 1.0], "description": "学习率"},
                "max_depth": {"type": "int", "default": 3, "range": [1, 10], "description": "最大深度"}
            },
            "requirements": ["scikit-learn"]
        },
        "mlp_regressor": {
            "name": "MLP回归",
            "class": "MLPRegressor",
            "module": "sklearn.neural_network",
            "description": "神经网络回归",
            "params": {
                "hidden_layer_sizes": {"type": "list", "default": [100, 50], "description": "隐藏层配置"},
                "activation": {"type": "choice", "default": "relu", "options": ["relu", "tanh", "logistic"],
                               "description": "激活函数"},
                "max_iter": {"type": "int", "default": 500, "range": [100, 2000], "description": "最大迭代次数"}
            },
            "requirements": ["scikit-learn"]
        }
    },

    # ==================== 聚类模型 ====================
    "clustering": {
        "kmeans": {
            "name": "K-Means",
            "class": "KMeans",
            "module": "sklearn.cluster",
            "description": "基于质心的聚类，适合球形分布",
            "params": {
                "n_clusters": {"type": "int", "default": 3, "range": [2, 20], "description": "聚类数量"},
                "init": {"type": "choice", "default": "k-means++", "options": ["k-means++", "random"],
                         "description": "初始化方法"},
                "n_init": {"type": "int", "default": 10, "range": [1, 50], "description": "初始化次数"}
            },
            "requirements": ["scikit-learn"]
        },
        "dbscan": {
            "name": "DBSCAN",
            "class": "DBSCAN",
            "module": "sklearn.cluster",
            "description": "基于密度的聚类，可识别噪声",
            "params": {
                "eps": {"type": "float", "default": 0.5, "range": [0.1, 2.0], "description": "邻域半径"},
                "min_samples": {"type": "int", "default": 5, "range": [2, 20], "description": "最小样本数"}
            },
            "requirements": ["scikit-learn"]
        },
        "agglomerative": {
            "name": "层次聚类",
            "class": "AgglomerativeClustering",
            "module": "sklearn.cluster",
            "description": "自底向上聚合，可生成树状图",
            "params": {
                "n_clusters": {"type": "int", "default": 3, "range": [2, 20], "description": "聚类数量"},
                "linkage": {"type": "choice", "default": "ward", "options": ["ward", "complete", "average", "single"],
                            "description": "链接方式"}
            },
            "requirements": ["scikit-learn"]
        },
        "birch": {
            "name": "BIRCH",
            "class": "Birch",
            "module": "sklearn.cluster",
            "description": "适合大数据集聚类",
            "params": {
                "n_clusters": {"type": "int", "default": 3, "range": [2, 20], "description": "聚类数量"},
                "threshold": {"type": "float", "default": 0.5, "range": [0.1, 2.0], "description": "聚类阈值"}
            },
            "requirements": ["scikit-learn"]
        },
        "gaussian_mixture": {
            "name": "高斯混合模型",
            "class": "GaussianMixture",
            "module": "sklearn.mixture",
            "description": "概率软聚类，可输出隶属度",
            "params": {
                "n_components": {"type": "int", "default": 3, "range": [2, 20], "description": "成分数量"},
                "covariance_type": {"type": "choice", "default": "full",
                                    "options": ["full", "tied", "diag", "spherical"], "description": "协方差类型"}
            },
            "requirements": ["scikit-learn"]
        },
        "optics": {
            "name": "OPTICS",
            "class": "OPTICS",
            "module": "sklearn.cluster",
            "description": "DBSCAN改进版，自动确定参数",
            "params": {
                "min_samples": {"type": "int", "default": 5, "range": [2, 20], "description": "最小样本数"},
                "xi": {"type": "float", "default": 0.05, "range": [0.01, 0.5], "description": "提取聚类的阈值"}
            },
            "requirements": ["scikit-learn"]
        },
        "spectral": {
            "name": "谱聚类",
            "class": "SpectralClustering",
            "module": "sklearn.cluster",
            "description": "基于图论的聚类",
            "params": {
                "n_clusters": {"type": "int", "default": 3, "range": [2, 20], "description": "聚类数量"},
                "affinity": {"type": "choice", "default": "rbf", "options": ["rbf", "nearest_neighbors"],
                             "description": "亲和力矩阵类型"}
            },
            "requirements": ["scikit-learn"]
        },
        "mean_shift": {
            "name": "均值漂移",
            "class": "MeanShift",
            "module": "sklearn.cluster",
            "description": "自动确定聚类数量",
            "params": {
                "bandwidth": {"type": "float", "default": None, "description": "带宽（None则自动估计）"}
            },
            "requirements": ["scikit-learn"]
        }
    },

    # ==================== 时间序列模型 ====================
    "time_series": {
        "arima": {
            "name": "ARIMA",
            "class": "ARIMAWrapper",
            "module": "autostat.models.arima_wrapper",
            "description": "自回归积分移动平均，经典时序模型",
            "params": {
                "p": {"type": "int", "default": 1, "range": [0, 5], "description": "自回归阶数 (AR)"},
                "d": {"type": "int", "default": 1, "range": [0, 2], "description": "差分阶数 (I)"},
                "q": {"type": "int", "default": 1, "range": [0, 5], "description": "移动平均阶数 (MA)"}
            },
            "requirements": ["statsmodels"]
        },
        "sarima": {
            "name": "SARIMA",
            "class": "SARIMAX",
            "module": "statsmodels.tsa.statespace.sarimax",
            "description": "季节性ARIMA，处理周期性数据",
            "params": {
                "p": {"type": "int", "default": 1, "range": [0, 5], "description": "自回归阶数"},
                "d": {"type": "int", "default": 1, "range": [0, 2], "description": "差分阶数"},
                "q": {"type": "int", "default": 1, "range": [0, 5], "description": "移动平均阶数"},
                "P": {"type": "int", "default": 1, "range": [0, 2], "description": "季节性自回归阶数"},
                "D": {"type": "int", "default": 1, "range": [0, 1], "description": "季节性差分阶数"},
                "Q": {"type": "int", "default": 1, "range": [0, 2], "description": "季节性移动平均阶数"},
                "s": {"type": "int", "default": 12, "range": [4, 52], "description": "季节周期"}
            },
            "requirements": ["statsmodels"]
        },
        "prophet": {
            "name": "Prophet",
            "class": "Prophet",
            "module": "prophet",
            "description": "Facebook时序预测，处理缺失值",
            "params": {
                "changepoint_prior_scale": {"type": "float", "default": 0.05, "range": [0.001, 0.5],
                                            "description": "变点先验尺度"},
                "seasonality_prior_scale": {"type": "float", "default": 10.0, "range": [0.01, 10.0],
                                            "description": "季节性先验尺度"},
                "holidays_prior_scale": {"type": "float", "default": 10.0, "range": [0.01, 10.0],
                                         "description": "节假日先验尺度"}
            },
            "requirements": ["prophet"]
        },
        "lstm_ts": {
            "name": "LSTM时序",
            "class": "LSTMTimeSeries",
            "module": "autostat.models.deep_learning",
            "description": "长短期记忆网络时序预测",
            "params": {
                "units": {"type": "int", "default": 50, "range": [16, 256], "description": "LSTM单元数"},
                "layers": {"type": "int", "default": 2, "range": [1, 4], "description": "LSTM层数"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 128], "description": "批次大小"},
                "lookback": {"type": "int", "default": 10, "range": [3, 30], "description": "回看窗口"}
            },
            "requirements": ["tensorflow"]
        },
        "gru_ts": {
            "name": "GRU时序",
            "class": "GRUTimeSeries",
            "module": "autostat.models.deep_learning",
            "description": "门控循环单元，轻量级LSTM",
            "params": {
                "units": {"type": "int", "default": 50, "range": [16, 256], "description": "GRU单元数"},
                "layers": {"type": "int", "default": 2, "range": [1, 4], "description": "GRU层数"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 128], "description": "批次大小"},
                "lookback": {"type": "int", "default": 10, "range": [3, 30], "description": "回看窗口"}
            },
            "requirements": ["tensorflow"]
        },
        "transformer_ts": {
            "name": "Transformer时序",
            "class": "TransformerTimeSeries",
            "module": "autostat.models.deep_learning",
            "description": "基于注意力机制的时序预测",
            "params": {
                "d_model": {"type": "int", "default": 64, "range": [32, 256], "description": "模型维度"},
                "n_heads": {"type": "int", "default": 4, "range": [2, 8], "description": "注意力头数"},
                "num_layers": {"type": "int", "default": 3, "range": [1, 6], "description": "Transformer层数"},
                "epochs": {"type": "int", "default": 50, "range": [10, 200], "description": "训练轮数"},
                "batch_size": {"type": "int", "default": 32, "range": [8, 128], "description": "批次大小"},
                "lookback": {"type": "int", "default": 20, "range": [5, 50], "description": "回看窗口"}
            },
            "requirements": ["tensorflow"]
        }
    }
}


class ModelRegistry:
    """模型注册表 - 管理所有模型信息"""

    @classmethod
    def get_task_types(cls):
        """获取所有任务类型"""
        return list(MODEL_REGISTRY.keys())

    @classmethod
    def get_models(cls, task_type: str):
        """获取指定任务类型的所有模型"""
        if task_type not in MODEL_REGISTRY:
            return {}
        return MODEL_REGISTRY[task_type]

    @classmethod
    def get_model(cls, task_type: str, model_key: str):
        """获取指定模型的信息"""
        if task_type not in MODEL_REGISTRY:
            return None
        return MODEL_REGISTRY[task_type].get(model_key)

    @classmethod
    def get_model_params(cls, task_type: str, model_key: str):
        """获取模型的参数配置"""
        model = cls.get_model(task_type, model_key)
        if not model:
            return {}
        return model.get("params", {})

    @classmethod
    def get_model_requirements(cls, task_type: str, model_key: str):
        """获取模型依赖"""
        model = cls.get_model(task_type, model_key)
        if not model:
            return []
        return model.get("requirements", [])

    @classmethod
    def check_requirements(cls, task_type: str, model_key: str):
        """检查模型依赖是否已安装"""
        requirements = cls.get_model_requirements(task_type, model_key)
        missing = []
        for req in requirements:
            try:
                __import__(req.replace("-", "_"))
            except ImportError:
                missing.append(req)
        return len(missing) == 0, missing

    @classmethod
    def get_model_class(cls, task_type: str, model_key: str):
        """动态导入模型类"""
        model = cls.get_model(task_type, model_key)
        if not model:
            return None

        module_name = model["module"]
        class_name = model["class"]

        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError:
            return None

    @classmethod
    def get_recommended_models(cls, task_type: str, data_shape: dict):
        """根据数据特征推荐模型"""
        n_samples = data_shape.get("n_samples", 0)
        n_features = data_shape.get("n_features", 0)
        n_classes = data_shape.get("n_classes", 0)

        models = cls.get_models(task_type)
        recommendations = []

        for key, info in models.items():
            score = 0
            reasons = []

            if task_type == "classification":
                if n_samples < 1000:
                    if key in ["logistic_regression", "naive_bayes", "svm"]:
                        score += 2
                        reasons.append("小数据量推荐")
                else:
                    if key in ["random_forest", "xgboost", "lightgbm", "catboost"]:
                        score += 2
                        reasons.append("大数据量推荐")

                if n_classes == 2:
                    if key in ["logistic_regression", "svm"]:
                        score += 1
                        reasons.append("二分类适用")
                else:
                    if key in ["random_forest", "xgboost"]:
                        score += 1
                        reasons.append("多分类适用")

                if n_features > 100:
                    if key in ["random_forest", "xgboost", "lightgbm"]:
                        score += 1
                        reasons.append("高维特征适用")

            elif task_type == "regression":
                if n_samples < 1000:
                    if key in ["linear_regression", "ridge", "lasso"]:
                        score += 2
                        reasons.append("小数据量推荐")
                else:
                    if key in ["random_forest_regressor", "xgboost_regressor", "lightgbm_regressor"]:
                        score += 2
                        reasons.append("大数据量推荐")

            elif task_type == "clustering":
                if n_samples < 1000:
                    if key in ["kmeans", "agglomerative"]:
                        score += 2
                        reasons.append("小数据量推荐")
                else:
                    if key in ["birch", "mini_batch_kmeans"]:
                        score += 2
                        reasons.append("大数据量推荐")

            recommendations.append({
                "key": key,
                "name": info["name"],
                "score": score,
                "reasons": reasons,
                "description": info["description"]
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]