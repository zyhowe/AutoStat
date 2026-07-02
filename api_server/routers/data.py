"""数据管理路由（完整版）"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.data_service import DataService
from api_server.services.session_service import SessionService
from api_server.services.database_service import DatabaseService
from api_server.schemas.data import DataPreviewResponse, DataUploadResponse

router = APIRouter()


class DatabaseLoadRequest(BaseModel):
    config: Dict[str, Any]
    table_names: List[str]
    limit: int = 5000
    max_text_length: int = 100
    relationships: Optional[List[Dict]] = None


class DatabaseLoadResponse(BaseModel):
    tables: Dict[str, Any]
    session_id: Optional[str] = None


@router.post("/data/upload", response_model=DataUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    data_service: DataService = Depends(Dependencies.get_data_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """上传数据文件"""
    file_path = Dependencies.save_upload_file(file)

    try:
        df = data_service.load_file(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件加载失败: {str(e)}")

    if session_id:
        session = session_service.get_session(session_id)
        if session:
            session_service.add_file(session_id, file.filename, str(file_path))

    variable_types = data_service.infer_types(df)

    return DataUploadResponse(
        file_name=file.filename,
        file_path=str(file_path),
        rows=len(df),
        columns=len(df.columns),
        variable_types=variable_types,
        preview=data_service.get_preview(df)
    )


@router.post("/data/database/load", response_model=DatabaseLoadResponse)
async def load_database(
        request: DatabaseLoadRequest,
        database_service: DatabaseService = Depends(Dependencies.get_database_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """从数据库加载表"""
    try:
        tables = database_service.load_tables(
            config=request.config,
            table_names=request.table_names,
            limit=request.limit,
            max_text_length=request.max_text_length,
            relationships=request.relationships
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据库加载失败: {str(e)}")

    if not tables:
        raise HTTPException(status_code=400, detail="没有成功加载任何表")

    source_name = f"{request.table_names[0]}_db"
    session_id = session_service.create_session(source_name, "database", {"tables": request.table_names})

    result = {}
    first_table = None

    for name, df in tables.items():
        if df is not None and not df.empty:
            variable_types = database_service.infer_types(df)
            result[name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "variable_types": variable_types,
                "preview": database_service.get_preview(df)
            }
            session_service.save_variable_types(session_id, variable_types)

            # 🆕 保存第一个表为临时CSV
            if first_table is None:
                first_table = (name, df)

    # 🆕 保存临时CSV文件
    if first_table:
        import tempfile
        name, df = first_table
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            df.to_csv(tmp.name, index=False)
            session_service.add_file(session_id, f"{name}.csv", tmp.name)
            print(f"✅ 保存临时CSV: {tmp.name}")

    return DatabaseLoadResponse(
        tables=result,
        session_id=session_id
    )

@router.post("/data/demo")
async def load_demo_data(
    dataset: str = "sales",
    data_service: DataService = Depends(Dependencies.get_data_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """加载示例数据"""
    import pandas as pd
    import numpy as np

    # 生成示例数据
    if dataset == "sales":
        np.random.seed(42)
        n = 5000
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "销售日期": np.random.choice(dates, n),
            "产品类别": np.random.choice(["电子产品", "服装", "食品", "家居", "图书"], n),
            "地区": np.random.choice(["华东", "华南", "华北", "西南", "西北"], n),
            "渠道": np.random.choice(["线上", "线下"], n, p=[0.6, 0.4]),
            "促销活动": np.random.choice(["双十一", "618", "平日促销", "无促销"], n),
            "销量": np.random.poisson(100, n),
            "单价": np.random.uniform(50, 500, n).round(2),
            "折扣": np.random.choice([0, 0.05, 0.1, 0.15, 0.2], n),
            "成本": np.random.uniform(30, 300, n).round(2)
        })
    elif dataset == "user":
        np.random.seed(42)
        n = 3000
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "用户ID": range(1, n + 1),
            "性别": np.random.choice(["男", "女"], n, p=[0.48, 0.52]),
            "年龄": np.random.normal(35, 12, n).astype(int),
            "城市": np.random.choice(["北京", "上海", "广州", "深圳", "成都", "武汉"], n),
            "会员等级": np.random.choice(["普通", "黄金", "铂金", "钻石"], n, p=[0.5, 0.3, 0.15, 0.05]),
            "设备类型": np.random.choice(["iOS", "Android", "Web"], n, p=[0.35, 0.45, 0.2]),
            "消费金额": np.random.exponential(500, n).round(2),
            "登录次数": np.random.poisson(10, n),
            "停留时长": np.random.exponential(300, n).round(0),
            "注册日期": dates,
        })
    elif dataset == "medical":
        np.random.seed(42)
        n = 2000
        ages = np.random.normal(55, 18, n).astype(int)
        ages = np.clip(ages, 18, 100)
        df = pd.DataFrame({
            "患者ID": range(1, n + 1),
            "性别": np.random.choice(["男", "女"], n),
            "年龄": ages,
            "疾病类型": np.random.choice(["高血压", "糖尿病", "冠心病", "肺炎", "骨折"], n),
            "收缩压": (110 + (ages - 50) * 0.5 + np.random.normal(0, 15, n)).round(0),
            "舒张压": (70 + (ages - 50) * 0.3 + np.random.normal(0, 10, n)).round(0),
            "心率": np.random.normal(75, 12, n).round(0),
            "血糖": (5.0 + (ages - 50) * 0.02 + np.random.normal(0, 1, n)).round(1),
            "是否吸烟": np.random.choice(["是", "否"], n, p=[0.3, 0.7]),
            "就诊日期": pd.date_range("2023-01-01", periods=n, freq="D"),
        })
    else:
        raise HTTPException(status_code=400, detail="不支持的示例数据集")

    # 创建会话
    source_name = f"{dataset}_demo"
    session_id = session_service.create_session(source_name, "single")

    # 保存文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
        df.to_csv(tmp.name, index=False)
        session_service.add_file(session_id, f"{dataset}_demo.csv", tmp.name)

    variable_types = data_service.infer_types(df)
    session_service.save_variable_types(session_id, variable_types)

    return {
        "session_id": session_id,
        "source_name": source_name,
        "rows": len(df),
        "columns": len(df.columns),
        "variable_types": variable_types,
        "preview": data_service.get_preview(df)
    }