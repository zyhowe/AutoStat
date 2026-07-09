import api from './index'

export const reportApi = {
  get(sessionId) {
    return api.get(`/report/${sessionId}`)
  },

  getSummary(sessionId) {
    return api.get(`/report/${sessionId}/summary`)
  },

  getInsights(sessionId) {
    return api.get(`/report/${sessionId}/insights`)
  },

  getQuality(sessionId) {
    return api.get(`/quality/${sessionId}`)
  },

  export(sessionId, format = 'html') {
    return api.get(`/export/${sessionId}`, {
      params: { format },
      responseType: 'blob'
    })
  },

  // 🆕 导出日志
  exportLog(sessionId) {
    return api.get(`/export/${sessionId}/log`, {
      responseType: 'blob'
    })
  },

  // ===== 新增：下载轻量 HTML 报告 =====
  async downloadHtml(sessionId) {
    // 获取 JSON 数据
    const jsonData = await this.get(sessionId)

    // 构建轻量 HTML
    const htmlContent = buildReportHtml(jsonData, sessionId)

    // 创建 Blob 并下载
    const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report_${sessionId}.html`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }
}

// ===== HTML 生成器 =====
function buildReportHtml(jsonData, sessionId) {
  // 提取数据
  const shape = jsonData.data_shape || { rows: 0, columns: 0 }
  const variableTypes = jsonData.variable_types || {}
  const summaries = jsonData.variable_summaries || {}
  const quality = jsonData.quality_report || {}
  const correlations = jsonData.correlations || {}
  const tsDiag = jsonData.time_series_diagnostics || {}
  const modelRecs = jsonData.model_recommendations || []
  const cleaning = jsonData.cleaning_suggestions || []
  const conclusions = jsonData.summary || []

  // 类型统计
  const typeCounts = {}
  Object.values(variableTypes).forEach(info => {
    const typ = info.type || info
    typeCounts[typ] = (typeCounts[typ] || 0) + 1
  })
  const typeDisplay = {
    continuous: '连续变量',
    categorical: '分类变量',
    categorical_numeric: '数值型分类',
    ordinal: '有序分类',
    datetime: '日期时间',
    identifier: '标识符',
    text: '文本'
  }

  // 强相关
  const highCorrs = correlations.high_correlations || []

  // 时序数据
  const tsData = Object.entries(tsDiag).map(([key, info]) => ({
    key,
    n_samples: info.n_samples || 0,
    is_stationary: info.is_stationary,
    has_autocorrelation: info.has_autocorrelation,
    has_seasonality: info.has_seasonality
  }))

  // 清理 JSON 数据（避免 script 注入）
  const safeJson = JSON.stringify(jsonData).replace(/<\/script/g, '<\\/script')

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoStat 分析报告</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"><\/script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f0f2f5;
            padding: 24px;
            color: #2c3e50;
        }
        .container { max-width: 1400px; margin: 0 auto; }

        /* 头部 */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 32px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .header .badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 12px;
            display: inline-block;
            margin-top: 12px;
        }

        /* 统计卡片 */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #2c3e50; }
        .stat-card .label { font-size: 13px; color: #909399; margin-top: 4px; }

        /* 卡片 */
        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 16px;
            border-left: 4px solid #667eea;
            padding-left: 16px;
        }
        .card h3 { font-size: 15px; margin: 16px 0 12px 0; color: #555; }

        /* 图表容器 */
        .chart-container {
            width: 100%;
            height: 300px;
            margin: 12px 0;
        }
        .chart-container-sm { height: 220px; }
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        /* 表格 */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th { background: #f8f9fa; font-weight: 600; }
        tr:hover { background: #f8f9fa; }

        /* 标签 */
        .tag {
            display: inline-block;
            padding: 2px 12px;
            border-radius: 12px;
            font-size: 11px;
        }
        .tag-high { background: #ff4757; color: white; }
        .tag-mid { background: #ffa502; color: white; }
        .tag-low { background: #2ed573; color: white; }
        .tag-info { background: #e3f2fd; color: #1565c0; }
        .tag-success { background: #e8f5e9; color: #2e7d32; }
        .tag-warning { background: #fff3e0; color: #e65100; }
        .tag-danger { background: #ffebee; color: #c62828; }

        /* 结论卡片 */
        .conclusion-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .conclusion-card {
            text-align: center;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #eee;
        }
        .conclusion-card .icon { font-size: 28px; }
        .conclusion-card .title { font-weight: 600; font-size: 14px; margin: 8px 0 4px; }
        .conclusion-card .desc { font-size: 12px; color: #909399; }

        /* 洞察列表 */
        .insight-list { padding-left: 20px; }
        .insight-list li { padding: 6px 0; line-height: 1.6; }

        /* 清洗建议 */
        .cleaning-item {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }

        /* 信息框 */
        .info-box {
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            font-size: 13px;
        }
        .info-box.blue { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .info-box.green { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .info-box.orange { background: #fff3e0; border-left: 4px solid #ff9800; }
        .info-box.red { background: #ffebee; border-left: 4px solid #f44336; }

        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

        .footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
            border-top: 1px solid #e9ecef;
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            .chart-row { grid-template-columns: 1fr; }
            .grid-2 { grid-template-columns: 1fr; }
            body { padding: 12px; }
            .header { padding: 20px; }
            .stats-row { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
<div class="container">
    <!-- 头部 -->
    <div class="header">
        <h1>📊 数据分析报告</h1>
        <p>生成时间: ${new Date().toLocaleString()}</p>
        <p>📁 数据源: ${jsonData.source_table || '未知'}</p>
        <div class="badge">🤖 AutoStat 智能分析报告</div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-row">
        <div class="stat-card"><div class="value">${shape.rows?.toLocaleString() || 0}</div><div class="label">总行数</div></div>
        <div class="stat-card"><div class="value">${shape.columns || 0}</div><div class="label">总列数</div></div>
        <div class="stat-card"><div class="value">${(quality.missing || []).length}</div><div class="label">缺失字段</div></div>
        <div class="stat-card"><div class="value">${quality.duplicates?.count || 0}</div><div class="label">重复记录</div></div>
    </div>

    <!-- ===== 核心结论 ===== -->
    <div class="card">
        <h2>🎯 核心结论</h2>
        ${conclusions.length > 0 ? `
        <div class="conclusion-grid">
            ${conclusions.map(c => `
            <div class="conclusion-card">
                <div class="icon">${c.icon || '📌'}</div>
                <div class="title">${c.title || ''}</div>
                <div class="desc">${c.description || ''}</div>
            </div>
            `).join('')}
        </div>
        ` : '<p style="color:#909399;">暂无核心结论</p>'}
    </div>

    <!-- ===== 变量类型分布（饼图） ===== -->
    <div class="card">
        <h2>📊 变量类型分布</h2>
        <div class="chart-row">
            <div id="typePieChart" class="chart-container chart-container-sm"></div>
            <div>
                <table>
                    <thead><tr><th>类型</th><th>数量</th></tr></thead>
                    <tbody>
                        ${Object.entries(typeCounts).map(([typ, count]) => `
                        <tr><td>${typeDisplay[typ] || typ}</td><td>${count}</td></tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ===== 数值变量统计 ===== -->
    <div class="card">
        <h2>📈 数值变量统计</h2>
        ${(() => {
            const contVars = Object.entries(summaries).filter(([name, info]) => info.type === 'continuous')
            if (contVars.length === 0) return '<p style="color:#909399;">无数值变量</p>'
            return `
            <div style="overflow-x:auto;">
            <table>
                <thead><tr><th>变量</th><th>样本量</th><th>缺失率</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小值</th><th>最大值</th></tr></thead>
                <tbody>
                    ${contVars.map(([name, info]) => `
                    <tr>
                        <td><strong>${name}</strong></td>
                        <td>${info.count || 0}</td>
                        <td>${(info.missing_pct || 0).toFixed(1)}%</td>
                        <td>${info.mean !== undefined ? info.mean.toFixed(2) : '-'}</td>
                        <td>${info.median !== undefined ? info.median.toFixed(2) : '-'}</td>
                        <td>${info.std !== undefined ? info.std.toFixed(2) : '-'}</td>
                        <td>${info.min !== undefined ? info.min.toFixed(2) : '-'}</td>
                        <td>${info.max !== undefined ? info.max.toFixed(2) : '-'}</td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
            </div>
            `
        })()}
    </div>

    <!-- ===== 分类变量统计 ===== -->
    <div class="card">
        <h2>🏷️ 分类变量统计</h2>
        ${(() => {
            const catVars = Object.entries(summaries).filter(([name, info]) =>
                ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
            )
            if (catVars.length === 0) return '<p style="color:#909399;">无分类变量</p>'
            return `
            <div style="overflow-x:auto;">
            <table>
                <thead><tr><th>变量</th><th>样本量</th><th>缺失率</th><th>类别数</th><th>众数</th><th>众数占比</th></tr></thead>
                <tbody>
                    ${catVars.map(([name, info]) => `
                    <tr>
                        <td><strong>${name}</strong></td>
                        <td>${info.count || 0}</td>
                        <td>${(info.missing_pct || 0).toFixed(1)}%</td>
                        <td>${info.n_unique || 0}</td>
                        <td>${info.mode || '-'}</td>
                        <td>${(info.mode_pct || 0).toFixed(1)}%</td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
            </div>
            `
        })()}
    </div>

    <!-- ===== 相关性热力图 ===== -->
    <div class="card">
        <h2>🔗 相关性热力图</h2>
        ${(() => {
            const matrix = correlations.matrix || {}
            const keys = Object.keys(matrix)
            if (keys.length < 2) return '<p style="color:#909399;">数值变量不足2个，无法生成热力图</p>'
            return `
            <div id="corrHeatmap" class="chart-container"></div>
            ${highCorrs.length > 0 ? `
            <div class="info-box blue">
                <strong>发现 ${highCorrs.length} 对强相关关系 (|r| > 0.7)</strong>
                <ul style="margin-top:8px; padding-left:20px;">
                    ${highCorrs.slice(0, 5).map(c => `
                    <li><strong>${c.var1}</strong> ↔ <strong>${c.var2}</strong>：r = ${c.value}</li>
                    `).join('')}
                    ${highCorrs.length > 5 ? `<li>... 还有 ${highCorrs.length - 5} 对</li>` : ''}
                </ul>
            </div>
            ` : ''}
            `
        })()}
    </div>

    <!-- ===== 时间序列诊断 ===== -->
    <div class="card">
        <h2>📈 时间序列分析</h2>
        ${tsData.length > 0 ? `
        <div style="overflow-x:auto;">
        <table>
            <thead><tr><th>变量/分组</th><th>样本量</th><th>平稳性</th><th>自相关性</th><th>季节性</th></tr></thead>
            <tbody>
                ${tsData.map(d => `
                <tr>
                    <td><strong>${d.key}</strong></td>
                    <td>${d.n_samples}</td>
                    <td><span class="tag ${d.is_stationary ? 'tag-success' : 'tag-warning'}">${d.is_stationary ? '✅ 平稳' : '⚠️ 非平稳'}</span></td>
                    <td><span class="tag ${d.has_autocorrelation ? 'tag-success' : 'tag-info'}">${d.has_autocorrelation ? '✅ 有' : '❌ 无'}</span></td>
                    <td><span class="tag ${d.has_seasonality ? 'tag-success' : 'tag-info'}">${d.has_seasonality ? '✅ 有' : '❌ 无'}</span></td>
                </tr>
                `).join('')}
            </tbody>
        </table>
        </div>
        ` : '<p style="color:#909399;">未检测到时间序列数据</p>'}
    </div>

    <!-- ===== 模型推荐 ===== -->
    <div class="card">
        <h2>🤖 模型推荐</h2>
        ${modelRecs.length > 0 ? `
        <div style="overflow-x:auto;">
        <table>
            <thead><tr><th>任务类型</th><th>目标</th><th>推荐模型</th><th>原因</th></tr></thead>
            <tbody>
                ${modelRecs.slice(0, 8).map(rec => `
                <tr>
                    <td><span class="tag ${rec.priority === '高' ? 'tag-high' : rec.priority === '中' ? 'tag-mid' : 'tag-low'}">${rec.priority || '中'}</span> ${rec.task_type || ''}</td>
                    <td>${rec.target_column || '-'}</td>
                    <td>${rec.ml || rec.traditional || '-'}</td>
                    <td style="font-size:12px; color:#666;">${rec.reason || ''}</td>
                </tr>
                `).join('')}
            </tbody>
        </table>
        </div>
        ` : '<p style="color:#909399;">暂无模型推荐</p>'}
    </div>

    <!-- ===== 勾稽规则 ===== -->
    <div class="card">
        <h2>🔗 勾稽规则</h2>
        ${(() => {
            const rules = quality.audit_rules || {}
            const allRules = [
                ...(rules.arithmetic_rules || []),
                ...(rules.functional_dependencies || []),
                ...(rules.temporal_rules || [])
            ]
            if (allRules.length === 0) return '<p style="color:#909399;">未发现勾稽规则</p>'
            return `
            <div style="overflow-x:auto;">
            <table>
                <thead><tr><th>规则</th><th>置信度</th><th>优先级</th><th>违反数</th></tr></thead>
                <tbody>
                    ${allRules.slice(0, 20).map(r => `
                    <tr>
                        <td style="font-family:monospace; font-size:12px;">${r.rule || ''}</td>
                        <td>${((r.confidence || 0) * 100).toFixed(1)}%</td>
                        <td><span class="tag ${r.priority === '高' ? 'tag-high' : r.priority === '中' ? 'tag-mid' : 'tag-low'}">${r.priority || '低'}</span></td>
                        <td>${r.violation_count || 0}</td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
            ${allRules.length > 20 ? `<p style="color:#909399; margin-top:8px;">... 还有 ${allRules.length - 20} 条规则</p>` : ''}
            </div>
            `
        })()}
    </div>

    <!-- ===== 清洗建议 ===== -->
    <div class="card">
        <h2>🧹 清洗建议</h2>
        ${cleaning.length > 0 ? `
        <ul class="insight-list">
            ${cleaning.map(s => `<li class="cleaning-item">${s}</li>`).join('')}
        </ul>
        ` : '<div class="info-box green">✅ 数据质量良好，无明显清洗需求</div>'}
    </div>

    <!-- ===== 页脚 ===== -->
    <div class="footer">
        <p>🤖 AutoStat 智能统计分析工具 | 报告自动生成 | 基于采样数据，结果可能存在误差</p>
        <p>📊 分析完成时间: ${new Date().toLocaleString()}</p>
    </div>
</div>

<!-- ===== ECharts 渲染脚本 ===== -->
<script>
(function() {
    const reportData = ${safeJson};

    // 1. 类型分布饼图
    const typePie = document.getElementById('typePieChart');
    if (typePie) {
        const chart = echarts.init(typePie);
        const typeCountsData = ${JSON.stringify(Object.entries(typeCounts).map(([typ, count]) => ({
            name: typeDisplay[typ] || typ,
            value: count
        })))};
        chart.setOption({
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            legend: { orient: 'vertical', left: 'left', top: 'center', itemWidth: 12, itemHeight: 12 },
            series: [{
                type: 'pie',
                radius: ['40%', '65%'],
                center: ['55%', '50%'],
                data: typeCountsData,
                label: { show: true, formatter: '{d}%', fontSize: 11 },
                labelLine: { show: true },
                emphasis: { scale: true }
            }]
        });
        window.addEventListener('resize', () => chart.resize());
    }

    // 2. 相关性热力图
    const corrChart = document.getElementById('corrHeatmap');
    if (corrChart) {
        const matrix = ${JSON.stringify(correlations.matrix || {})};
        const keys = Object.keys(matrix);
        if (keys.length >= 2) {
            const chart = echarts.init(corrChart);
            const data = [];
            for (let i = 0; i < keys.length; i++) {
                for (let j = 0; j < keys.length; j++) {
                    const val = matrix[keys[i]]?.[keys[j]];
                    if (val !== undefined && val !== null && !isNaN(val)) {
                        data.push([i, j, parseFloat(val.toFixed(2))]);
                    } else {
                        data.push([i, j, null]);
                    }
                }
            }
            chart.setOption({
                tooltip: {
                    position: 'top',
                    formatter: function(params) {
                        const d = params.data;
                        if (d && d.length >= 3 && d[2] !== null) {
                            return keys[d[0]] + ' ↔ ' + keys[d[1]] + '<br/>相关系数：' + d[2];
                        }
                        return '';
                    }
                },
                grid: { left: '12%', right: '8%', top: '5%', bottom: '12%' },
                xAxis: {
                    type: 'category',
                    data: keys,
                    splitArea: { show: true },
                    axisLabel: { rotate: 30, fontSize: 10, interval: 0 }
                },
                yAxis: {
                    type: 'category',
                    data: keys,
                    splitArea: { show: true },
                    axisLabel: { fontSize: 10 }
                },
                visualMap: {
                    min: -1,
                    max: 1,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: 0,
                    inRange: { color: ['#FFFFFF', '#F56C6C', '#409EFF'] },
                    outOfRange: { color: '#FFFFFF' }
                },
                series: [{
                    type: 'heatmap',
                    data: data,
                    label: {
                        show: true,
                        fontSize: 9,
                        formatter: function(p) {
                            return p.data[2] !== null && p.data[2] !== undefined ? p.data[2] : '';
                        }
                    },
                    itemStyle: {
                        color: function(params) {
                            if (params.data[2] === null || params.data[2] === undefined) {
                                return '#FFFFFF';
                            }
                            const val = params.data[2];
                            if (val > 0.7) return '#F56C6C';
                            if (val < -0.7) return '#409EFF';
                            if (val > 0.3) return '#E6A23C';
                            if (val < -0.3) return '#67C23A';
                            return '#FFFFFF';
                        }
                    }
                }]
            });
            window.addEventListener('resize', () => chart.resize());
        }
    }
})();
<\/script>
</body>
</html>`
}