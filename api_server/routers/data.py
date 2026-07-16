"""数据管理路由（完整版）"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.data_service import DataService
from api_server.services.session_service import SessionService
from api_server.services.database_service import DatabaseService
from api_server.schemas.data import DataPreviewResponse, DataUploadResponse
from api_server.routers.session import get_client_ip
from api_server.schemas.data import (
    DatabaseLoadRequest, DatabaseLoadResponse,
    TableInfo, CandidateRelation,
    RelationConfirmRequest, RelationConfirmResponse
)

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
    table_list: List[TableInfo] = []
    candidate_relations: List[CandidateRelation] = []
    load_summary: Dict[str, Any] = {}


class RelationConfirmRequest(BaseModel):
    session_id: str
    relationships: List[Dict]


class RelationConfirmResponse(BaseModel):
    success: bool
    message: str


# ==================== 数据预览/筛选接口 ====================

class FilterCondition(BaseModel):
    """筛选条件"""
    field: str
    condition: str  # eq, gt, lt, gte, lte, between, contains, is_null, is_not_null, is_outlier, expr
    value: Any = None


class DataPreviewRequest(BaseModel):
    session_id: str
    filters: List[FilterCondition] = []
    fields: Optional[List[str]] = None
    page: int = 1
    page_size: int = 100


class DataPreviewResponse(BaseModel):
    total: int
    page: int
    page_size: int
    rows: List[Dict[str, Any]]
    columns: List[str]
    filter_desc: str


@router.post("/data/preview", response_model=DataPreviewResponse)
async def preview_data(
    request: DataPreviewRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service),
    data_service: DataService = Depends(Dependencies.get_data_service)
):
    """
    预览/筛选数据

    支持条件:
    - eq: 等于
    - gt: 大于
    - lt: 小于
    - gte: 大于等于
    - lte: 小于等于
    - between: 区间 [min, max]
    - contains: 包含字符串
    - is_null: 为空
    - is_not_null: 不为空
    - is_outlier: 异常值（基于 IQR）
    - expr: 表达式（如 "A != B" 或 "abs(A - B) > 0.01"）
    """
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取数据路径（优先 Parquet）
    data_path = session_service.get_data_path(request.session_id)
    if not data_path:
        raise HTTPException(status_code=400, detail="会话没有关联的数据文件")

    df = data_service.load_file(data_path)

    # 应用筛选条件
    filter_desc_parts = []
    for f in request.filters:
        df, desc = _apply_filter(df, f)
        if desc:
            filter_desc_parts.append(desc)

    total_rows = len(df)

    # 字段选择
    if request.fields:
        available_fields = [col for col in request.fields if col in df.columns]
        if available_fields:
            df = df[available_fields]

    columns = df.columns.tolist()

    # 分页
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    page_df = df.iloc[start:end]

    return DataPreviewResponse(
        total=total_rows,
        page=request.page,
        page_size=request.page_size,
        rows=page_df.to_dict(orient="records"),
        columns=columns,
        filter_desc="; ".join(filter_desc_parts) if filter_desc_parts else "无筛选条件"
    )


def _apply_filter(df: Any, filter_cond: FilterCondition):
    """应用单个筛选条件，返回 (过滤后的df, 描述文本)"""
    import pandas as pd
    import numpy as np

    field = filter_cond.field
    cond = filter_cond.condition
    value = filter_cond.value

    # ==================== 表达式条件（用于勾稽规则违反） ====================
    if cond == "expr":
        # value 是一个表达式字符串，如 "A != B" 或 "abs(A - B) > 0.01"
        try:
            # 使用 pandas eval 计算布尔掩码
            mask = df.eval(value)
            df = df[mask]
            desc = f" 满足表达式: {value}"
            return df, desc
        except Exception as e:
            print(f"⚠️ 表达式过滤失败: {e}")
            return df, f" 表达式无效: {value}"

    if field not in df.columns:
        return df, None

    desc = f"{field}"

    if cond == "eq":
        df = df[df[field] == value]
        desc += f" = {value}"
    elif cond == "gt":
        df = df[df[field] > value]
        desc += f" > {value}"
    elif cond == "lt":
        df = df[df[field] < value]
        desc += f" < {value}"
    elif cond == "gte":
        df = df[df[field] >= value]
        desc += f" >= {value}"
    elif cond == "lte":
        df = df[df[field] <= value]
        desc += f" <= {value}"
    elif cond == "between":
        if isinstance(value, list) and len(value) == 2:
            min_val, max_val = value[0], value[1]
            df = df[(df[field] >= min_val) & (df[field] <= max_val)]
            desc += f" 在 [{min_val}, {max_val}] 之间"
    elif cond == "contains":
        if value:
            df = df[df[field].astype(str).str.contains(str(value), na=False)]
            desc += f" 包含 '{value}'"
    elif cond == "is_null":
        df = df[df[field].isna()]
        desc += " 为空"
    elif cond == "is_not_null":
        df = df[df[field].notna()]
        desc += " 不为空"
    elif cond == "is_outlier":
        series = df[field].dropna()
        if len(series) > 0:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[field] < lower) | (df[field] > upper)]
                desc += " 异常值"
            else:
                df = df.iloc[0:0]
                desc += " 异常值 (无)"
        else:
            df = df.iloc[0:0]
            desc += " 异常值 (无数据)"
    else:
        return df, None

    return df, desc


# ==================== 原有路由 ====================

@router.post("/data/upload", response_model=DataUploadResponse)
async def upload_file(
    request: Request,
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

    if session_id:
        parquet_path = session_service.get_data_parquet_path(session_id)
        data_service.save_to_parquet(df, parquet_path)

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
    request: Request,
    request_body: DatabaseLoadRequest,
    database_service: DatabaseService = Depends(Dependencies.get_database_service),
    session_service: SessionService = Depends(Dependencies.get_session_service),
    data_service: DataService = Depends(Dependencies.get_data_service)
):
    """从数据库加载表（支持多表）"""
    try:
        tables = database_service.load_tables(
            config=request_body.config,
            table_names=request_body.table_names,
            limit=request_body.limit,
            max_text_length=request_body.max_text_length,
            relationships=request_body.relationships
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据库加载失败: {str(e)}")

    if not tables:
        raise HTTPException(status_code=400, detail="没有成功加载任何表")

    # 过滤空表
    valid_tables = {name: df for name, df in tables.items() if df is not None and not df.empty}

    if not valid_tables:
        raise HTTPException(status_code=400, detail="所有表均为空")

    # 创建会话
    source_name = f"{request_body.table_names[0]}_db" if request_body.table_names else "database"
    if len(request_body.table_names) > 1:
        source_name = f"{source_name}_multi"
    client_ip = get_client_ip(request)
    session_service.set_client_ip(client_ip)
    session_id = session_service.create_session(
        source_name,
        "database",
        {"tables": request_body.table_names, "db_config": request_body.config}
    )

    table_list = []
    result = {}

    # 保存所有表到 Parquet
    for name, df in valid_tables.items():
        variable_types = database_service.infer_types(df)
        preview = database_service.get_preview(df)

        table_info = TableInfo(
            name=name,
            rows=len(df),
            columns=len(df.columns),
            preview=preview,
            variable_types=variable_types,
            load_status="success"
        )
        table_list.append(table_info)
        result[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "variable_types": variable_types,
            "preview": preview
        }

        session_service.save_variable_types(session_id, variable_types)

    # ✅ 保存所有表到 Parquet
    save_results = data_service.save_tables_to_parquet(valid_tables, session_service, session_id)

    # ✅ 自动识别表间关系
    candidate_relations = []
    if len(valid_tables) >= 2:
        try:
            from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
            # 创建多表分析器
            analyzer = MultiTableStatisticalAnalyzer(valid_tables)
            # 获取自动发现的关系
            discovered = analyzer.discovered_relationships
            all_rels = analyzer.all_relationships
            if all_rels and 'foreign_keys' in all_rels:
                for fk in all_rels['foreign_keys']:
                    candidate_relations.append(CandidateRelation(
                        from_table=fk.get('from_table', ''),
                        from_col=fk.get('from_col', ''),
                        to_table=fk.get('to_table', ''),
                        to_col=fk.get('to_col', ''),
                        relation_type=fk.get('type', 'many_to_one'),
                        confidence=fk.get('confidence', 0.5),
                        auto_discovered=fk.get('auto_discovered', True)
                    ))
            print(f"✅ 自动发现 {len(candidate_relations)} 条候选关系")
        except Exception as e:
            print(f"⚠️ 关系自动发现失败: {e}")

    # 保存到 metadata
    session_service.save_relationships(session_id, [r.dict() for r in candidate_relations])

    # 保存临时CSV（兼容旧版）
    if valid_tables:
        first_name, first_df = next(iter(valid_tables.items()))
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            first_df.to_csv(tmp.name, index=False)
            session_service.add_file(session_id, f"{first_name}.csv", tmp.name)

    return DatabaseLoadResponse(
        tables=result,
        session_id=session_id,
        table_list=table_list,
        candidate_relations=candidate_relations,
        load_summary={
            "total": len(valid_tables),
            "success": len([t for t in table_list if t.load_status == "success"]),
            "failed": len([t for t in table_list if t.load_status == "failed"])
        }
    )


@router.post("/data/relations/confirm")
async def confirm_relations(
    request: RelationConfirmRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """确认表间关系"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    session_service.save_relationships(request.session_id, request.relationships)

    return RelationConfirmResponse(
        success=True,
        message=f"已确认 {len(request.relationships)} 条关系"
    )


@router.get("/data/relations/{session_id}")
async def get_relations(
    session_id: str,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取表间关系"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    relationships = session_service.get_relationships(session_id)
    return {"session_id": session_id, "relationships": relationships}


@router.post("/data/demo")
async def load_demo_data(
    request: Request,
    dataset: str = "sales",
    data_service: DataService = Depends(Dependencies.get_data_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """加载示例数据"""
    import pandas as pd
    import numpy as np

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

    source_name = f"{dataset}_demo"
    client_ip = get_client_ip(request)
    session_service.set_client_ip(client_ip)
    session_id = session_service.create_session(source_name, "single")

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
        df.to_csv(tmp.name, index=False)
        session_service.add_file(session_id, f"{dataset}_demo.csv", tmp.name)

    parquet_path = session_service.get_data_parquet_path(session_id)
    data_service.save_to_parquet(df, parquet_path)

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