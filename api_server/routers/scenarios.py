"""
场景路由
场景推导和场景仪表板 API
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.tech_fact_sheet import build_tech_fact_sheet
from api_server.services.scenario_engine import derive_scenarios
from api_server.services.scenario_executor import execute_scenarios
from api_server.services.llm_translator import translate_scenarios, parse_field_mapping
from api_server.services.config_service import ConfigService
from api_server.services.data_service import DataService
from api_server.services.insight_analyzer import analyze_insights

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Schemas =====

class ScenarioUpdateRequest(BaseModel):
    """场景更新请求"""
    scenarios: List[dict]


class ScenarioExecuteRequest(BaseModel):
    """场景执行请求"""
    scenario_ids: Optional[List[str]] = None


class TranslateRequest(BaseModel):
    """翻译请求"""
    field_mapping: Dict[str, str]  # {字段名: 中文名}


class ParseMappingRequest(BaseModel):
    """解析字段映射请求"""
    text: str  # 用户输入的任意格式文本


class ParseMappingResponse(BaseModel):
    """解析字段映射响应"""
    mapping: Dict[str, str]  # {字段名: 中文名}
    unmatched: List[str]     # 未识别的字段名


class SaveMappingRequest(BaseModel):
    """保存字段映射请求"""
    field_mapping: Dict[str, str]


# ===== API 端点 =====

@router.get("/scenarios/{session_id}")
async def get_scenarios(
    session_id: str,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """
    获取场景列表

    如果 scenarios.json 已存在，直接读取
    如果不存在，从 meta.json 推导候选场景
    """
    # 检查是否已执行过场景推导
    scenarios_data = session_service.load_scenarios(session_id)

    if scenarios_data:
        return {
            "session_id": session_id,
            "status": scenarios_data.get("status", "executed"),
            "candidates": scenarios_data.get("candidates", []),
            "results": scenarios_data.get("results", []),
            "has_results": True,
            "field_mapping": scenarios_data.get("field_mapping", {}),
            "insights": scenarios_data.get("insights", {})
        }

    # 未执行推导，从 meta.json 构建候选场景
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="分析尚未完成，请先执行分析")

    # 构建技术事实清单
    fact_sheet = build_tech_fact_sheet(session_id, analysis_result)

    # 推导候选场景
    candidates = derive_scenarios(fact_sheet)

    return {
        "session_id": session_id,
        "status": "draft",
        "candidates": candidates,
        "results": [],
        "has_results": False,
        "tech_fact_sheet": fact_sheet,
        "field_mapping": {},
        "insights": {}
    }


@router.post("/scenarios/{session_id}/update")
async def update_scenarios(
    session_id: str,
    request: ScenarioUpdateRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """更新场景列表（用户编辑后保存）"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取现有数据以保留 field_mapping 和 insights
    existing_data = session_service.load_scenarios(session_id) or {}

    # 保存场景配置
    session_service.save_scenarios(session_id, {
        "status": "draft",
        "candidates": request.scenarios,
        "results": existing_data.get("results", []),
        "field_mapping": existing_data.get("field_mapping", {}),
        "insights": existing_data.get("insights", {})
    })

    return {
        "session_id": session_id,
        "message": f"已更新 {len(request.scenarios)} 个场景",
        "count": len(request.scenarios)
    }


@router.post("/scenarios/{session_id}/mapping/save")
async def save_mapping(
    session_id: str,
    request: SaveMappingRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """保存字段映射"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取现有数据
    existing_data = session_service.load_scenarios(session_id) or {}

    # 更新映射
    session_service.save_scenarios(session_id, {
        "status": existing_data.get("status", "draft"),
        "candidates": existing_data.get("candidates", []),
        "results": existing_data.get("results", []),
        "field_mapping": request.field_mapping,
        "insights": existing_data.get("insights", {})
    })

    return {
        "session_id": session_id,
        "message": f"已保存 {len(request.field_mapping)} 个字段映射",
        "count": len(request.field_mapping)
    }


@router.post("/scenarios/{session_id}/execute")
async def execute_scenarios_api(
    session_id: str,
    request: ScenarioExecuteRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service),
    data_service: DataService = Depends(Dependencies.get_data_service)
):
    """执行场景深度计算"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取场景配置
    scenarios_data = session_service.load_scenarios(session_id)
    if not scenarios_data:
        raise HTTPException(status_code=400, detail="请先推导场景")

    candidates = scenarios_data.get("candidates", [])
    field_mapping = scenarios_data.get("field_mapping", {})

    # 过滤要执行的场景
    if request.scenario_ids:
        to_execute = [s for s in candidates if s.get("id") in request.scenario_ids and s.get("enabled", True)]
    else:
        to_execute = [s for s in candidates if s.get("enabled", True)]

    if not to_execute:
        raise HTTPException(status_code=400, detail="没有可执行的场景")

    # 加载数据
    data_path = session_service.get_data_path(session_id)
    if not data_path:
        raise HTTPException(status_code=400, detail="没有可用的数据")

    df = data_service.load_file(data_path)

    # 构建技术事实清单
    analysis_result = session.get("analysis_result", {})
    fact_sheet = build_tech_fact_sheet(session_id, analysis_result)

    # 执行场景
    results = execute_scenarios(df, fact_sheet, to_execute, field_mapping)

    # ===== 执行洞察分析 =====
    insights = {}
    try:
        insights = analyze_insights(df, results, fact_sheet, field_mapping)
        logger.info(f"洞察分析完成: {len(insights)} 个维度")
    except Exception as e:
        logger.error(f"洞察分析失败: {e}")
        import traceback
        traceback.print_exc()
        # 洞察分析失败不影响场景执行结果

    # 保存结果
    session_service.save_scenarios(session_id, {
        "status": "executed",
        "candidates": candidates,
        "results": results,
        "field_mapping": field_mapping,
        "insights": insights,
        "executed_at": __import__('datetime').datetime.now().isoformat()
    })

    return {
        "session_id": session_id,
        "message": f"成功执行 {len(results)} 个场景",
        "results": results,
        "count": len(results)
    }


@router.get("/scenarios/{session_id}/insights")
async def get_insights(
    session_id: str,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取洞察分析数据"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    scenarios_data = session_service.load_scenarios(session_id)
    if not scenarios_data:
        return {
            "session_id": session_id,
            "has_insights": False,
            "insights": {},
            "message": "请先执行场景"
        }

    insights = scenarios_data.get("insights", {})
    if not insights:
        return {
            "session_id": session_id,
            "has_insights": False,
            "insights": {},
            "message": "暂无洞察数据，请重新执行场景"
        }

    return {
        "session_id": session_id,
        "has_insights": True,
        "insights": insights,
        "generated_at": insights.get("generated_at")
    }


@router.post("/scenarios/{session_id}/mapping/parse")
async def parse_field_mapping_api(
    session_id: str,
    request: ParseMappingRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service),
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """
    解析用户输入的字段映射文本
    大模型从任意格式文本中抽取 字段名 ↔ 中文名 映射
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not request.text or not request.text.strip():
        return ParseMappingResponse(mapping={}, unmatched=[])

    # 获取大模型配置
    llm_configs = config_service.get_llm_configs()
    llm_config = llm_configs[0] if llm_configs else None

    if not llm_config:
        # 无大模型配置，尝试简单规则匹配
        mapping, unmatched = parse_field_mapping_with_rules(request.text)
        return ParseMappingResponse(mapping=mapping, unmatched=unmatched)

    # 用大模型解析
    try:
        mapping, unmatched = parse_field_mapping(request.text, llm_config)
        return ParseMappingResponse(mapping=mapping, unmatched=unmatched)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")


@router.post("/scenarios/{session_id}/translate")
async def translate_scenarios_api(
    session_id: str,
    request: TranslateRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service),
    config_service: ConfigService = Depends(Dependencies.get_config_service),
    data_service: DataService = Depends(Dependencies.get_data_service)
):
    """
    大模型翻译场景结果
    需要先确认字段映射
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取场景结果
    scenarios_data = session_service.load_scenarios(session_id)
    if not scenarios_data or scenarios_data.get("status") != "executed":
        raise HTTPException(status_code=400, detail="请先执行场景")

    results = scenarios_data.get("results", [])

    if not results:
        raise HTTPException(status_code=400, detail="没有可翻译的场景结果")

    # 获取大模型配置
    llm_configs = config_service.get_llm_configs()
    llm_config = llm_configs[0] if llm_configs else None

    if not llm_config:
        raise HTTPException(status_code=400, detail="未配置大模型，请先在设置中配置")

    # 获取字段映射
    field_mapping = request.field_mapping or {}

    # 构建表结构
    analysis_result = session.get("analysis_result", {})
    variable_types = analysis_result.get("variable_types", {})
    summaries = analysis_result.get("variable_summaries", {})

    table_structure = {
        "fields": [
            {
                "name": col,
                "type": info.get("type") if isinstance(info, dict) else info,
                "sample": summaries.get(col, {}).get("mode", "")
            }
            for col, info in list(variable_types.items())[:30]
        ]
    }

    # 翻译
    translated = translate_scenarios(results, table_structure, field_mapping, llm_config)

    # 保存翻译结果
    scenarios_data["results"] = translated
    scenarios_data["translated_at"] = __import__('datetime').datetime.now().isoformat()
    scenarios_data["field_mapping"] = field_mapping
    session_service.save_scenarios(session_id, scenarios_data)

    return {
        "session_id": session_id,
        "message": f"已翻译 {len(translated)} 个场景",
        "results": translated
    }


@router.get("/scenarios/{session_id}/dashboard")
async def get_dashboard(
    session_id: str,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取场景仪表板数据"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    scenarios_data = session_service.load_scenarios(session_id)

    if not scenarios_data or scenarios_data.get("status") != "executed":
        analysis_result = session.get("analysis_result", {})
        return {
            "session_id": session_id,
            "has_results": False,
            "data_overview": {
                "rows": analysis_result.get("data_shape", {}).get("rows", 0),
                "columns": analysis_result.get("data_shape", {}).get("columns", 0),
                "quality_score": analysis_result.get("quality_report", {}).get("overall_score", 0)
            },
            "scenarios": [],
            "message": "请先执行场景推导"
        }

    results = scenarios_data.get("results", [])
    field_mapping = scenarios_data.get("field_mapping", {})

    return {
        "session_id": session_id,
        "has_results": True,
        "data_overview": {
            "rows": session.get("data_shape", {}).get("rows", 0),
            "columns": session.get("data_shape", {}).get("columns", 0),
            "quality_score": session.get("quality_report", {}).get("overall_score", 0)
        },
        "scenarios": results,
        "summary": {
            "total": len(results),
            "completed": len([r for r in results if r.get("status") == "completed"]),
            "failed": len([r for r in results if r.get("status") == "failed"])
        },
        "field_mapping": field_mapping
    }


# ===== 辅助函数 =====

def parse_field_mapping_with_rules(text: str):
    """简单规则匹配（无大模型时的降级方案）"""
    mapping = {}
    unmatched = []

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 尝试多种分隔符
        separators = ['：', ':', '=', '→', '->', '，', ',']
        for sep in separators:
            if sep in line:
                parts = line.split(sep, 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    mapping[key] = value
                    break
        else:
            # 尝试用空格分隔
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = ' '.join(parts[1:])

    return mapping, []