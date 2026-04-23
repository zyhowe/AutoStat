"""
自动训练服务 - 分析完成后自动训练所有推荐模型
"""

import streamlit as st
import time
import traceback
from typing import Dict, Any, Optional, List

from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.model_training_service import (
    get_model_recommendations_from_json,
    get_models_by_task_type,
    get_model_params,
    execute_training
)
from web.services.feature_flags import FeatureFlags


def auto_train_from_recommendation(session_id: str) -> List[Dict]:
    """
    根据所有推荐自动训练模型（依次训练，失败继续）

    返回: 训练结果列表
    """
    results = []

    # 使用容器显示训练状态
    train_container = st.container()

    with train_container:
        st.markdown("### 🤖 自动训练进行中")

        try:
            # 加载分析结果
            json_data = StorageService.load_json("analysis_result", session_id)
            if not json_data:
                st.warning("自动训练：未找到分析结果")
                return results

            # 获取推荐列表
            all_recommendations = get_model_recommendations_from_json(json_data)
            if not all_recommendations:
                st.info("自动训练：无模型推荐")
                return results

            # 加载数据（只加载一次，复用）
            data = StorageService.load_dataframe("processed_data", session_id)
            if data is None:
                st.warning("自动训练：未找到数据")
                return results

            total = len(all_recommendations)
            st.progress(0, text=f"共 {total} 个模型待训练")

            print(f"自动训练: 开始，共 {total} 个模型")
            print(f"自动训练: 数据形状 {data.shape}")

            # 依次训练每个推荐
            for idx, rec in enumerate(all_recommendations):
                st.markdown(f"---")
                st.markdown(f"### 模型 {idx+1}/{total}")

                # 显示推荐内容
                task_type_str = rec.get('task_type', '')
                target_col = rec.get('target_column', '')
                feature_cols = rec.get('feature_columns', [])

                print(f"自动训练: 模型 {idx+1} - 原始任务类型: {task_type_str}, 目标: {target_col}, 特征数: {len(feature_cols)}")

                st.caption(f"任务类型: {task_type_str}")
                st.caption(f"目标列: {target_col if target_col else '无'}")
                st.caption(f"特征列数: {len(feature_cols)}")

                # ========== 根据任务类型选择实际模型 ==========
                task_type = None
                model_key = None

                # 1. 时间序列预测 → 使用 ARIMA
                if "时间序列预测" in task_type_str:
                    task_type = "time_series"
                    model_key = "arima"  # statsmodels 的 ARIMA
                    print(f"自动训练: 时间序列预测 -> 使用 ARIMA")
                    st.caption("📅 时间序列预测 -> 使用 ARIMA 模型")

                    # 时间序列需要特殊处理：特征列应该是目标列本身
                    if target_col:
                        valid_features = [target_col]
                    else:
                        valid_features = []

                # 2. 聚类分析 → 使用 K-Means
                elif "聚类分析" in task_type_str:
                    task_type = "clustering"
                    model_key = "kmeans"  # sklearn 的 K-Means
                    target_col = None
                    print(f"自动训练: 聚类分析 -> 使用 K-Means")
                    st.caption("🔘 聚类分析 -> 使用 K-Means 模型")

                    # 验证特征列存在
                    valid_features = []
                    missing_features = []
                    for f in feature_cols:
                        if f in data.columns:
                            valid_features.append(f)
                        else:
                            missing_features.append(f)

                    if missing_features:
                        print(f"自动训练: 缺失特征列 {missing_features}")
                        st.warning(f"部分特征列不存在: {missing_features}")

                # 3. 回归预测 → 使用随机森林
                elif "回归预测" in task_type_str:
                    task_type = "regression"
                    model_key = "random_forest_regressor"
                    print(f"自动训练: 回归预测 -> 使用随机森林回归")
                    st.caption("📈 回归预测 -> 使用随机森林回归模型")

                    # 验证特征列存在
                    valid_features = []
                    missing_features = []
                    for f in feature_cols:
                        if f in data.columns:
                            valid_features.append(f)
                        else:
                            missing_features.append(f)

                    if missing_features:
                        print(f"自动训练: 缺失特征列 {missing_features}")
                        st.warning(f"部分特征列不存在: {missing_features}")

                # 4. 分类预测 → 使用随机森林
                elif "分类预测" in task_type_str:
                    task_type = "classification"
                    model_key = "random_forest"
                    print(f"自动训练: 分类预测 -> 使用随机森林分类")
                    st.caption("📊 分类预测 -> 使用随机森林分类模型")

                    # 验证特征列存在
                    valid_features = []
                    missing_features = []
                    for f in feature_cols:
                        if f in data.columns:
                            valid_features.append(f)
                        else:
                            missing_features.append(f)

                    if missing_features:
                        print(f"自动训练: 缺失特征列 {missing_features}")
                        st.warning(f"部分特征列不存在: {missing_features}")

                else:
                    st.warning(f"跳过：无法识别任务类型 {task_type_str}")
                    results.append({"success": False, "error": f"无法识别任务类型: {task_type_str}", "recommendation": rec})
                    st.progress((idx + 1) / total, text=f"进度: {idx+1}/{total}")
                    continue

                # 检查有效特征
                if not valid_features and task_type != "time_series":
                    st.warning(f"跳过：没有有效特征列")
                    results.append({"success": False, "error": "没有有效特征列", "recommendation": rec})
                    st.progress((idx + 1) / total, text=f"进度: {idx+1}/{total}")
                    continue

                # 检查目标列（分类/回归需要）
                if task_type in ["classification", "regression"] and (not target_col or target_col not in data.columns):
                    st.warning(f"跳过：目标列 {target_col} 不存在")
                    results.append({"success": False, "error": f"目标列 {target_col} 不存在", "recommendation": rec})
                    st.progress((idx + 1) / total, text=f"进度: {idx+1}/{total}")
                    continue

                # 获取模型参数
                model_params = get_model_params(task_type, model_key) if model_key else {}
                default_params = {}
                for param_name, param_info in model_params.items():
                    default_params[param_name] = param_info.get('default')

                # 生成模型名称
                from web.services.model_training_service import generate_model_name
                user_model_name = f"自动训练_{generate_model_name(task_type, target_col if target_col else '', model_key)}"

                print(f"自动训练: 开始训练 {user_model_name} (task_type={task_type}, model_key={model_key}, features={len(valid_features)})")

                # 显示训练状态
                status_placeholder = st.empty()
                status_placeholder.info(f"🔄 正在训练: {user_model_name}...")

                # 执行训练
                try:
                    success, result = execute_training(
                        data, valid_features, target_col, task_type,
                        model_key, default_params,
                        train_ratio=0.7, val_ratio=0.15,
                        missing_strategy="drop", scaling="standard", encoding="onehot",
                        cv_folds=5, random_seed=42, session_id=session_id,
                        user_model_name=user_model_name
                    )

                    if success:
                        status_placeholder.success(f"✅ 训练完成: {user_model_name}")
                        results.append({"success": True, "model_name": user_model_name, "result": result})
                        print(f"自动训练: 成功 - {user_model_name}")
                    else:
                        error_msg = result.get('error', '未知错误')
                        status_placeholder.error(f"❌ 训练失败: {error_msg}")
                        print(f"自动训练: 失败 - {error_msg}")
                        traceback_msg = result.get('traceback', '')
                        if traceback_msg:
                            print(traceback_msg)
                        results.append({"success": False, "error": error_msg, "recommendation": rec})

                except Exception as e:
                    error_msg = str(e)
                    status_placeholder.error(f"❌ 训练异常: {error_msg}")
                    print(f"自动训练: 异常 - {error_msg}")
                    print(traceback.format_exc())
                    results.append({"success": False, "error": error_msg, "recommendation": rec})

                # 更新进度
                st.progress((idx + 1) / total, text=f"进度: {idx+1}/{total} 已完成")

                # 短暂停顿
                time.sleep(0.5)

            # 汇总结果
            st.markdown("---")
            success_count = sum(1 for r in results if r.get("success"))
            fail_count = total - success_count
            st.markdown(f"### 📊 自动训练完成")
            st.markdown(f"✅ 成功: {success_count} 个")
            st.markdown(f"❌ 失败: {fail_count} 个")

            if fail_count > 0:
                st.markdown("**失败的模型:**")
                for r in results:
                    if not r.get("success"):
                        st.caption(f"  - {r.get('recommendation', {}).get('task_type', '未知')}: {r.get('error', '未知错误')}")

            print(f"自动训练: 完成 - 成功 {success_count}, 失败 {fail_count}")

        except Exception as e:
            error_msg = str(e)
            st.error(f"自动训练出错: {error_msg}")
            print(f"自动训练出错: {error_msg}")
            print(traceback.format_exc())

    return results