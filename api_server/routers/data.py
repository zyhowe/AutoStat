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
    RelationConfirmRequest, RelationConfirmResponse,
    FieldTypesUpdateRequest, FieldTypesUpdateResponse  # 新增
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


# ==================== 统一的多表关系发现和会话创建 ====================

def _discover_relations_and_create_session(
    tables: Dict[str, Any],
    source_name: str,
    source_type: str,
    request: Request,
    data_service: DataService,
    session_service: SessionService,
    session_id: Optional[str] = None,
    db_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    统一的多表关系发现和会话创建

    参数:
    - tables: 表名字典 {表名: DataFrame}
    - source_name: 数据源名称
    - source_type: 来源类型 (multi_upload / database / demo)
    - request: FastAPI Request
    - data_service: DataService 实例
    - session_service: SessionService 实例
    - session_id: 可选，如果传入则复用已有会话
    - db_config: 可选的数据库配置信息

    返回:
    - 统一格式的响应字典
    """
    import pandas as pd

    # ===== 1. 创建或复用会话 =====
    client_ip = get_client_ip(request)
    session_service.set_client_ip(client_ip)

    if session_id:
        # 复用已有会话
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    else:
        # 创建新会话
        tables_info = {"tables": list(tables.keys()), "source": source_type}
        if db_config:
            tables_info["db_config"] = db_config
        session_id = session_service.create_session(source_name, "database", tables_info)

    # ===== 2. 保存所有表到 Parquet =====
    table_list = []
    result = {}
    all_variable_types = {}

    for name, df in tables.items():
        variable_types = data_service.infer_types(df)
        preview = data_service.get_preview(df)

        table_info = {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "preview": preview,
            "variable_types": variable_types,
            "load_status": "success"
        }
        table_list.append(table_info)
        result[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "variable_types": variable_types,
            "preview": preview
        }
        all_variable_types[name] = variable_types

        # 保存到 Parquet
        data_service.save_to_parquet(df, session_service, session_id, name)
        session_service.save_table_info(session_id, name, {
            "rows": len(df),
            "columns": len(df.columns),
            "saved_at": pd.Timestamp.now().isoformat()
        })

    # 保存 variable_types
    for name, vt in all_variable_types.items():
        session_service.save_variable_types(session_id, vt)

    # ===== 3. 统一发现关系 =====
    candidate_relations = []
    if len(tables) >= 2:
        try:
            from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
            analyzer = MultiTableStatisticalAnalyzer(tables)
            all_rels = analyzer.discover_relationships_only()
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
            print(f"✅ {source_type} 发现 {len(candidate_relations)} 条候选关系")
        except Exception as e:
            print(f"⚠️ {source_type} 关系发现失败: {e}")
            # 即使失败也继续，允许用户手动添加
            import traceback
            traceback.print_exc()

    # 保存关系到 session（即使为空也保存）
    session_service.save_relationships(session_id, [r.dict() for r in candidate_relations])

    return {
        "tables": result,
        "session_id": session_id,
        "table_list": table_list,
        "candidate_relations": [r.dict() for r in candidate_relations],  # 可能为空
        "load_summary": {
            "total": len(tables),
            "success": len([t for t in table_list if t["load_status"] == "success"]),
            "failed": len([t for t in table_list if t["load_status"] == "failed"])
        }
    }


# ==================== 文件上传 ====================

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
        # 标准签名：save_to_parquet(df, session_service, session_id, table_name)
        data_service.save_to_parquet(df, session_service, session_id, "data")

    return DataUploadResponse(
        file_name=file.filename,
        file_path=str(file_path),
        rows=len(df),
        columns=len(df.columns),
        variable_types=variable_types,
        preview=data_service.get_preview(df)
    )


# ==================== 数据库加载 ====================

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

    source_name = f"{request_body.table_names[0]}_db" if request_body.table_names else "database"
    if len(request_body.table_names) > 1:
        source_name = f"{source_name}_multi"

    # 统一调用，传入 session_id=None 创建新会话
    return _discover_relations_and_create_session(
        tables=valid_tables,
        source_name=source_name,
        source_type="database",
        request=request,
        data_service=data_service,
        session_service=session_service,
        session_id=None,
        db_config=request_body.config
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


# ==================== 新增：字段类型更新 ====================

@router.post("/data/field_types/update", response_model=FieldTypesUpdateResponse)
async def update_field_types(
    request: FieldTypesUpdateRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """
    更新字段类型缓存

    用户在预览界面调整字段类型后，调用此接口保存到后端缓存
    分析时从缓存读取字段类型
    """
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 保存到缓存
    success = session_service.save_field_types_cache(
        request.session_id,
        request.table_name,
        request.field_types
    )

    if not success:
        raise HTTPException(status_code=500, detail="保存字段类型失败")

    return FieldTypesUpdateResponse(
        success=True,
        message=f"字段类型已更新，共 {len(request.field_types)} 个字段"
    )


# ==================== 示例数据 ====================

def _handle_single_table_demo(df, dataset, request, data_service, session_service):
    """处理单表示例数据"""
    source_name = f"{dataset}_demo"
    client_ip = get_client_ip(request)
    session_service.set_client_ip(client_ip)
    session_id = session_service.create_session(source_name, "single")

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
        df.to_csv(tmp.name, index=False)
        session_service.add_file(session_id, f"{dataset}_demo.csv", tmp.name)

    data_service.save_to_parquet(df, session_service, session_id, "demo")

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


def _handle_multi_table_ecommerce(request, data_service, session_service):
    """
    生成电商多表示例数据（订单、客户、订单明细）
    统一使用 discover_relationships_only() 发现关系
    """
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # ----- 1. 客户表 (200条) -----
    n_customers = 200
    customers = pd.DataFrame({
        "id": range(1, n_customers + 1),
        "name": [f"客户_{i}" for i in range(1, n_customers + 1)],
        "city": np.random.choice(["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"], n_customers),
        "member_level": np.random.choice(["普通", "黄金", "铂金", "钻石"], n_customers, p=[0.5, 0.3, 0.15, 0.05]),
        "register_date": pd.date_range("2023-01-01", periods=n_customers, freq="D"),
        "age": np.random.normal(32, 10, n_customers).astype(int)
    })
    customers["age"] = customers["age"].clip(18, 70)

    # ----- 2. 订单表 (500条) -----
    n_orders = 500
    order_dates = pd.date_range("2023-06-01", periods=n_orders, freq="D")
    customer_ids = customers["id"].values

    orders = pd.DataFrame({
        "id": range(1, n_orders + 1),
        "customer_id": np.random.choice(customer_ids, n_orders),
        "order_date": order_dates,
        "status": np.random.choice(["已完成", "已发货", "已付款", "已取消"], n_orders, p=[0.6, 0.2, 0.15, 0.05]),
        "total_amount": np.random.uniform(50, 5000, n_orders).round(2),
        "payment_method": np.random.choice(["支付宝", "微信支付", "银行卡", "货到付款"], n_orders, p=[0.4, 0.35, 0.15, 0.1])
    })

    # ----- 3. 订单明细表 (1200条) -----
    n_items = 1200
    products = ["iPhone 15", "MacBook Pro", "AirPods Pro", "iPad Air", "Apple Watch",
                "机械键盘", "电竞鼠标", "4K显示器", "USB-C Hub", "移动硬盘",
                "咖啡机", "空气炸锅", "扫地机器人", "智能音箱", "投影仪"]

    order_ids = orders["id"].values

    order_items = pd.DataFrame({
        "id": range(1, n_items + 1),
        "order_id": np.random.choice(order_ids, n_items),
        "product_name": np.random.choice(products, n_items),
        "quantity": np.random.randint(1, 5, n_items),
        "unit_price": np.random.uniform(10, 2000, n_items).round(2)
    })

    # 计算明细金额
    order_items["subtotal"] = (order_items["quantity"] * order_items["unit_price"]).round(2)

    # ----- 4. 构建返回数据 -----
    tables = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items
    }

    # 统一调用 _discover_relations_and_create_session
    return _discover_relations_and_create_session(
        tables=tables,
        source_name="ecommerce_demo",
        source_type="demo",
        request=request,
        data_service=data_service,
        session_service=session_service,
        session_id=None
    )


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
        return _handle_single_table_demo(df, dataset, request, data_service, session_service)

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
        return _handle_single_table_demo(df, dataset, request, data_service, session_service)

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
        return _handle_single_table_demo(df, dataset, request, data_service, session_service)

    elif dataset == "ecommerce":
        return _handle_multi_table_ecommerce(request, data_service, session_service)

    else:
        raise HTTPException(status_code=400, detail="不支持的示例数据集")


# ==================== 多文件上传 ====================

@router.post("/data/upload/multi")
async def upload_multiple_files(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    data_service: DataService = Depends(Dependencies.get_data_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """
    多文件上传 - 每个文件作为一张表
    支持 CSV, Excel, JSON, TXT
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    # 限制文件数量
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="一次最多上传50个文件")

    tables = {}
    for file in files:
        # 保存上传文件到临时路径
        file_path = Dependencies.save_upload_file(file)

        # 加载 DataFrame
        try:
            df = data_service.load_file(str(file_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 加载失败: {str(e)}")

        if df.empty:
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 为空")

        # 使用文件名（不含扩展名）作为表名
        import os
        base_name = os.path.splitext(file.filename)[0]
        # 如果有重名，加后缀
        if base_name in tables:
            idx = 2
            while f"{base_name}_{idx}" in tables:
                idx += 1
            base_name = f"{base_name}_{idx}"

        tables[base_name] = df

    source_name = f"multi_upload_{len(tables)}tables" if not session_id else None

    # 统一调用 _discover_relations_and_create_session
    return _discover_relations_and_create_session(
        tables=tables,
        source_name=source_name or "multi_upload",
        source_type="multi_upload",
        request=request,
        data_service=data_service,
        session_service=session_service,
        session_id=session_id
    )