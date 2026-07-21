from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_server.routers import session, data, analysis, quality, report, export, models, chat, config
from api_server.routers.scenarios import router as scenarios_router
from api_server.config import settings

app = FastAPI(
    title="AutoStat API",
    description="AutoStat 智能统计分析工具 API",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册所有路由
app.include_router(session.router, prefix="/api/v1", tags=["会话管理"])
app.include_router(data.router, prefix="/api/v1", tags=["数据管理"])
app.include_router(analysis.router, prefix="/api/v1", tags=["分析执行"])
app.include_router(quality.router, prefix="/api/v1", tags=["质量报告"])
app.include_router(report.router, prefix="/api/v1", tags=["分析报告"])
app.include_router(export.router, prefix="/api/v1", tags=["导出"])
app.include_router(models.router, prefix="/api/v1", tags=["模型管理"])
app.include_router(chat.router, prefix="/api/v1", tags=["AI对话"])
app.include_router(config.router, prefix="/api/v1", tags=["配置管理"])
app.include_router(scenarios_router, prefix="/api/v1", tags=["场景管理"])


@app.get("/")
async def root():
    return {"name": "AutoStat API", "version": "0.2.0", "docs": "/api/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )