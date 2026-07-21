<template>
  <div class="analysis-summary">
    <!-- 无结果 -->
    <div v-if="!hasResults" class="empty-results">
      <el-empty description="暂无分析结果，请先在「开始分析」执行场景">
        <el-button type="primary" @click="$emit('go-to-config')">去执行场景</el-button>
      </el-empty>
    </div>

    <!-- 有结果 -->
    <div v-else class="insight-results">
      <!-- 统计卡片 -->
      <div class="stats-row">
        <div class="stat-card"><div class="stat-value">{{ scenarioResults.length }}</div><div class="stat-label">场景总数</div></div>
        <div class="stat-card"><div class="stat-value">{{ summary.completed || 0 }}</div><div class="stat-label">已完成</div></div>
        <div class="stat-card"><div class="stat-value">{{ summary.failed || 0 }}</div><div class="stat-label">失败</div></div>
        <div class="stat-card" v-if="totalRecords > 0"><div class="stat-value">{{ totalRecords }}</div><div class="stat-label">明细记录数</div></div>
      </div>

      <!-- 洞察摘要 -->
      <div v-if="hasInsights" class="insights-section">
        <!-- 排行榜区域 -->
        <div class="insights-grid">
          <!-- 规则违反排行榜 -->
          <div class="insight-card" v-if="insights.rankings?.violation_by_rule?.length > 0">
            <div class="card-title">📊 规则违反排行榜</div>
            <el-table :data="insights.rankings.violation_by_rule.slice(0, 8)" size="small" max-height="280">
              <el-table-column prop="name" label="规则" min-width="160" show-overflow-tooltip />
              <el-table-column prop="count" label="违反数" width="80" align="center" sortable />
              <el-table-column prop="rate" label="占比" width="70" align="center">
                <template #default="{ row }">{{ row.rate }}%</template>
              </el-table-column>
              <el-table-column label="严重程度" width="80" align="center">
                <template #default="{ row }">
                  <el-tag :type="row.severity === 'high' ? 'danger' : row.severity === 'medium' ? 'warning' : 'info'" size="small">
                    {{ row.severity }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column label="操作" width="70" align="center">
                <template #default="{ row }">
                  <el-button size="small" text type="primary" @click="viewByRule(row.name)">查看</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>

          <!-- 公司违反排行榜 -->
          <div class="insight-card" v-if="insights.rankings?.violation_by_company?.length > 0">
            <div class="card-title">🏢 公司违反排行榜</div>
            <el-table :data="insights.rankings.violation_by_company.slice(0, 8)" size="small" max-height="280">
              <el-table-column prop="name" label="公司" min-width="120" show-overflow-tooltip />
              <el-table-column prop="count" label="违反数" width="80" align="center" sortable />
              <el-table-column prop="rate" label="占比" width="70" align="center">
                <template #default="{ row }">{{ row.rate }}%</template>
              </el-table-column>
              <el-table-column label="操作" width="70" align="center">
                <template #default="{ row }">
                  <el-button size="small" text type="primary" @click="viewByCompany(row.name)">查看</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>

          <!-- 字段异常排行榜 -->
          <div class="insight-card" v-if="insights.rankings?.outlier_by_field?.length > 0">
            <div class="card-title">⚠️ 字段异常排行榜</div>
            <el-table :data="insights.rankings.outlier_by_field.slice(0, 8)" size="small" max-height="280">
              <el-table-column prop="name" label="字段" min-width="120" show-overflow-tooltip />
              <el-table-column prop="count" label="异常数" width="80" align="center" sortable />
              <el-table-column prop="rate" label="异常率" width="70" align="center">
                <template #default="{ row }">{{ row.rate }}%</template>
              </el-table-column>
              <el-table-column label="操作" width="70" align="center">
                <template #default="{ row }">
                  <el-button size="small" text type="primary" @click="viewByField(row.field || row.name)">查看</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>

          <!-- 公司异常排行榜 -->
          <div class="insight-card" v-if="insights.rankings?.outlier_by_company?.length > 0">
            <div class="card-title">🏢 公司异常排行榜</div>
            <el-table :data="insights.rankings.outlier_by_company.slice(0, 8)" size="small" max-height="280">
              <el-table-column prop="name" label="公司" min-width="120" show-overflow-tooltip />
              <el-table-column prop="count" label="异常数" width="80" align="center" sortable />
              <el-table-column prop="rate" label="占比" width="70" align="center">
                <template #default="{ row }">{{ row.rate }}%</template>
              </el-table-column>
              <el-table-column label="操作" width="70" align="center">
                <template #default="{ row }">
                  <el-button size="small" text type="primary" @click="viewByCompany(row.name)">查看</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>

        <!-- 趋势 -->
        <div class="trends-grid" v-if="hasTrends">
          <div class="insight-card" v-if="insights.trends?.violation_by_year?.length > 1">
            <div class="card-title">📈 违反趋势（按年）</div>
            <v-chart :option="violationTrendOption" class="trend-chart" autoresize />
          </div>
          <div class="insight-card" v-if="insights.trends?.outlier_by_year?.length > 1">
            <div class="card-title">📈 异常趋势（按年）</div>
            <v-chart :option="outlierTrendOption" class="trend-chart" autoresize />
          </div>
        </div>

        <!-- 集中度 -->
        <div class="concentration-card" v-if="insights.concentration?.level">
          <div class="card-title">📊 集中度分析</div>
          <el-descriptions :column="3" border size="small">
            <el-descriptions-item label="HHI">{{ insights.concentration.hhi }}</el-descriptions-item>
            <el-descriptions-item label="CR4">{{ insights.concentration.cr4 }}%</el-descriptions-item>
            <el-descriptions-item label="集中度等级">
              <el-tag :type="insights.concentration.level === '高度集中' ? 'danger' : insights.concentration.level === '中度集中' ? 'warning' : 'success'" size="small">
                {{ insights.concentration.level }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>
          <div v-if="insights.concentration.top_entities?.length > 0" style="margin-top: 8px;">
            <span style="font-size: 12px; color: #909399;">Top实体：</span>
            <el-tag v-for="item in insights.concentration.top_entities.slice(0, 5)" :key="item.name" size="small" style="margin: 2px;">
              {{ item.name }} ({{ item.rate }}%)
            </el-tag>
          </div>
        </div>

        <!-- 群体画像 -->
        <div class="cluster-card" v-if="insights.clusters?.has_clusters">
          <div class="card-title">👥 群体画像</div>
          <el-table :data="insights.clusters.profile" border size="small">
            <el-table-column prop="cluster_id" label="群体" width="80" align="center" />
            <el-table-column prop="size" label="样本数" width="100" align="center" />
            <el-table-column prop="rate" label="占比" width="80" align="center">
              <template #default="{ row }">{{ row.rate }}%</template>
            </el-table-column>
            <el-table-column prop="features" label="核心特征" min-width="200">
              <template #default="{ row }">
                <span v-if="row.features && row.features.length > 0">{{ row.features.join('; ') }}</span>
                <span v-else>—</span>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <!-- 冗余特征 -->
        <div class="correlation-card" v-if="insights.correlations?.redundant_pairs?.length > 0">
          <div class="card-title">🔗 冗余特征 (|r| >= 0.95)</div>
          <el-table :data="insights.correlations.redundant_pairs.slice(0, 10)" border size="small" max-height="250">
            <el-table-column prop="var1" label="变量1" min-width="120" />
            <el-table-column prop="var2" label="变量2" min-width="120" />
            <el-table-column prop="value" label="相关系数" width="120" align="center" />
            <el-table-column prop="type" label="类型" width="100" align="center">
              <template #default="{ row }">
                <el-tag :type="row.type === '完全冗余' ? 'danger' : 'warning'" size="small">{{ row.type }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
          <div style="margin-top: 8px; font-size: 12px; color: #909399;">
            💡 建议：完全冗余特征可考虑删除其中之一，高度冗余特征可考虑特征选择
          </div>
        </div>
      </div>

      <!-- 场景卡片 -->
      <div class="scenario-cards" style="margin-top: 24px;">
        <div
          v-for="scenario in scenarioResults"
          :key="scenario.scenario_id"
          class="scenario-card"
          :class="scenario.status"
        >
          <div class="card-header">
            <div class="header-left">
              <span class="card-status">{{ scenario.status === 'completed' ? '✅' : '❌' }}</span>
              <span class="card-name">{{ scenario.business_name || scenario.name }}</span>
              <el-tag v-if="scenario.business_summary" size="small" type="success">已解读</el-tag>
              <span class="record-count" v-if="getRecordCount(scenario) > 0">
                📋 {{ getRecordCount(scenario) }} 条明细
              </span>
            </div>
            <div class="header-right">
              <el-button size="small" text @click="$emit('toggle-expand', scenario.scenario_id)">
                {{ expandedCard === scenario.scenario_id ? '收起' : '展开' }}
              </el-button>
            </div>
          </div>

          <div class="card-body">
            <div class="card-left">
              <div class="panel-label">📊 技术结论</div>
              <div class="tech-items">
                <div v-for="(conclusion, idx) in scenario.conclusions" :key="idx" class="tech-item">
                  <span class="tech-icon">{{ getConclusionIcon(conclusion.type) }}</span>
                  <span class="tech-text">{{ replaceFieldNames(conclusion.text) }}</span>
                  <span v-if="conclusion.confidence" class="tech-confidence">{{ (conclusion.confidence * 100).toFixed(0) }}%</span>
                </div>
              </div>
              <div class="card-footer-actions" v-if="getRecordCount(scenario) > 0">
                <el-button size="small" type="primary" @click="$emit('view-details', scenario)">
                  📋 查看明细 →
                </el-button>
              </div>
            </div>

            <div class="card-right">
              <div class="panel-label">💡 业务解读</div>
              <div v-if="!scenario.business_summary" class="translate-placeholder">
                <span>暂无业务解读</span>
                <span class="hint">请先配置字段映射并重新执行</span>
              </div>
              <div v-else class="business-content">
                <div class="business-summary-text">
                  <span class="summary-icon">📌</span>
                  <span>{{ scenario.business_summary }}</span>
                </div>
                <div v-if="scenario.business_findings && scenario.business_findings.length > 0" class="business-findings">
                  <div class="findings-title">🔍 详细发现</div>
                  <ul><li v-for="(item, idx) in scenario.business_findings" :key="idx">{{ item }}</li></ul>
                </div>
                <div v-if="scenario.business_actions && scenario.business_actions.length > 0" class="business-actions">
                  <div class="actions-title">📋 建议行动</div>
                  <ul><li v-for="(item, idx) in scenario.business_actions" :key="idx">{{ item }}</li></ul>
                </div>
              </div>
              <div v-if="scenario._translation_error" class="business-error">
                <el-alert :title="'翻译失败: ' + scenario._translation_error" type="warning" show-icon :closable="false" />
              </div>
            </div>
          </div>

          <div v-if="expandedCard === scenario.scenario_id" class="card-detail">
            <el-divider />
            <pre>{{ JSON.stringify(scenario, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'

const props = defineProps({
  hasResults: { type: Boolean, default: false },
  scenarioResults: { type: Array, default: () => [] },
  summary: { type: Object, default: () => ({ total: 0, completed: 0, failed: 0 }) },
  totalRecords: { type: Number, default: 0 },
  fieldMapping: { type: Object, default: () => ({}) },
  insights: { type: Object, default: () => ({}) },
  loadingInsights: { type: Boolean, default: false },
  expandedCard: { type: [String, null], default: null }
})

const emit = defineEmits(['view-details', 'toggle-expand', 'go-to-config'])

const hasInsights = computed(() => {
  const i = props.insights
  return i && Object.keys(i).length > 0 && i.rankings
})

const hasTrends = computed(() => {
  const i = props.insights
  return i?.trends && (
    (i.trends.violation_by_year?.length > 1) ||
    (i.trends.outlier_by_year?.length > 1)
  )
})

const violationTrendOption = computed(() => {
  const data = props.insights?.trends?.violation_by_year || []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '15%' },
    xAxis: { type: 'category', data: data.map(d => d.period) },
    yAxis: { type: 'value', name: '违反数' },
    series: [{
      type: 'line',
      data: data.map(d => d.count),
      smooth: true,
      lineStyle: { color: '#f56c6c', width: 2 },
      areaStyle: { color: 'rgba(245,108,108,0.2)' },
      symbol: 'circle',
      symbolSize: 6
    }]
  }
})

const outlierTrendOption = computed(() => {
  const data = props.insights?.trends?.outlier_by_year || []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '15%' },
    xAxis: { type: 'category', data: data.map(d => d.period) },
    yAxis: { type: 'value', name: '异常数' },
    series: [{
      type: 'line',
      data: data.map(d => d.count),
      smooth: true,
      lineStyle: { color: '#e6a23c', width: 2 },
      areaStyle: { color: 'rgba(230,162,60,0.2)' },
      symbol: 'circle',
      symbolSize: 6
    }]
  }
})

function getRecordCount(scenario) {
  return (scenario.records || []).length
}

function getConclusionIcon(type) {
  const map = { 'summary': '📌', 'detail': '📋', 'alert': '⚠️', 'success': '✅', 'error': '❌' }
  return map[type] || '📌'
}

function replaceFieldNames(text) {
  if (!text) return text
  let result = text
  for (const [key, value] of Object.entries(props.fieldMapping)) {
    result = result.replace(new RegExp(key, 'g'), value)
  }
  return result
}

function viewByRule(rule) {
  // 找到对应场景并跳转
  const scenario = props.scenarioResults.find(s => {
    const records = s.records || []
    return records.some(r => r.rule === rule)
  })
  if (scenario) {
    emit('view-details', scenario)
  }
}

function viewByCompany(company) {
  // 找到包含该公司记录的场景
  // 由于公司不在records里直接存储，需要从row反查
  // 这里简化处理：跳转到数据解码并提示
  emit('view-details', { name: company, business_name: company, scenario_id: null })
}

function viewByField(field) {
  // 找到对应场景
  const scenario = props.scenarioResults.find(s => {
    const records = s.records || []
    return records.some(r => r.field === field || r.field_display === field)
  })
  if (scenario) {
    emit('view-details', scenario)
  }
}
</script>

<style scoped>
.analysis-summary { padding: 10px 0; }

.empty-results { padding: 60px 0; }

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}
.stat-card { background: #f5f7fa; border-radius: 8px; padding: 12px; text-align: center; }
.stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
.stat-label { font-size: 12px; color: #909399; }

.insights-section { margin-bottom: 24px; }
.insights-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}
.insight-card {
  background: #fff;
  border-radius: 10px;
  border: 1px solid #e4e7ed;
  padding: 14px 16px;
}
.card-title { font-size: 14px; font-weight: 600; color: #2c3e50; margin-bottom: 10px; }

.trends-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}
.trend-chart { width: 100%; height: 200px; }

.concentration-card,
.cluster-card,
.correlation-card {
  background: #fff;
  border-radius: 10px;
  border: 1px solid #e4e7ed;
  padding: 14px 16px;
  margin-bottom: 16px;
}

.scenario-cards { display: flex; flex-direction: column; gap: 16px; }
.scenario-card {
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  overflow: hidden;
  background: #fff;
}
.scenario-card.failed { border-left: 4px solid #f56c6c; }
.scenario-card:not(.failed) { border-left: 4px solid #67c23a; }

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 18px;
  background: #f8f9fa;
  border-bottom: 1px solid #e4e7ed;
}
.header-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.card-status { font-size: 16px; }
.card-name { font-size: 15px; font-weight: 600; color: #2c3e50; }
.record-count { font-size: 12px; color: #409eff; }

.card-body { display: flex; min-height: 200px; }
.card-left { flex: 1; padding: 16px 18px; border-right: 1px solid #e4e7ed; background: #fafafa; }
.card-right { flex: 1; padding: 16px 18px; background: #fff; }
.panel-label { font-size: 12px; font-weight: 600; color: #909399; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }

.tech-items { display: flex; flex-direction: column; gap: 6px; }
.tech-item { display: flex; align-items: flex-start; gap: 8px; padding: 4px 0; font-size: 13px; color: #555; line-height: 1.5; }
.tech-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.tech-text { flex: 1; word-break: break-word; }
.tech-confidence { font-size: 11px; color: #909399; flex-shrink: 0; }

.translate-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 140px; color: #909399; font-size: 14px; gap: 4px; }
.translate-placeholder .hint { font-size: 12px; color: #bbb; }

.business-content { display: flex; flex-direction: column; gap: 10px; }
.business-summary-text { display: flex; align-items: flex-start; gap: 10px; padding: 10px 14px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #409eff; }
.summary-icon { font-size: 18px; flex-shrink: 0; }
.business-summary-text span:last-child { font-size: 15px; font-weight: 500; color: #2c3e50; }

.business-findings, .business-actions { padding: 0 4px; }
.findings-title, .actions-title { font-size: 13px; font-weight: 600; color: #2c3e50; margin-bottom: 4px; }
.business-findings ul, .business-actions ul { margin: 0; padding-left: 20px; }
.business-findings li, .business-actions li { font-size: 13px; color: #555; line-height: 1.6; padding: 2px 0; }

.business-error { margin-top: 8px; }
.card-footer-actions { margin-top: 12px; }
.card-detail { padding: 16px 18px; background: #f8f9fa; }
.card-detail pre { background: #fff; padding: 12px; border-radius: 4px; font-size: 12px; max-height: 300px; overflow: auto; border: 1px solid #e4e7ed; }

@media (max-width: 992px) {
  .insights-grid { grid-template-columns: 1fr; }
  .trends-grid { grid-template-columns: 1fr; }
  .card-body { flex-direction: column; }
  .card-left { border-right: none; border-bottom: 1px solid #e4e7ed; }
}
@media (max-width: 768px) {
  .stats-row { grid-template-columns: repeat(2, 1fr); }
}
</style>