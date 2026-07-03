<template>
  <div class="report-page">
    <div class="report-header">
      <h2>📄 分析报告</h2>
      <div class="report-actions">
        <el-button size="small" @click="expandAll">📂 全部展开</el-button>
        <el-button size="small" @click="collapseAll">📁 全部折叠</el-button>
        <el-button size="small" type="primary" plain @click="handleExport('html')">📄 导出HTML</el-button>
        <el-button size="small" type="success" plain @click="handleExport('json')">📋 导出JSON</el-button>
        <el-button size="small" type="info" plain @click="handleExportLog">📝 导出日志</el-button>
      </div>
    </div>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="15" animated />
    </div>

    <div v-else-if="summary.length > 0 || reportData" class="report-content">
      <!-- 核心结论 -->
      <div class="conclusions-section">
        <h3>🎯 核心结论</h3>
        <div class="conclusion-cards">
          <el-card
            v-for="(conclusion, index) in summary"
            :key="index"
            class="conclusion-card"
            shadow="hover"
          >
            <div class="conclusion-icon">{{ conclusion.icon || '📌' }}</div>
            <div class="conclusion-title">{{ conclusion.title }}</div>
            <div class="conclusion-desc">{{ conclusion.description }}</div>
          </el-card>
        </div>
      </div>

      <!-- 智能解读 -->
      <div class="insights-section">
        <h3>💡 智能解读</h3>
        <el-card shadow="hover">
          <el-timeline>
            <el-timeline-item
              v-for="(finding, index) in insightFindings"
              :key="index"
              :type="index === 0 ? 'primary' : 'info'"
            >
              {{ finding }}
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </div>

      <!-- 详细分析（折叠） -->
      <div class="details-section">
        <h3>📋 详细分析</h3>

        <el-collapse v-model="activeNames">
          <!-- 数据概览 -->
          <el-collapse-item name="overview">
            <template #title>
              <span>📊 数据概览</span>
            </template>
            <div class="detail-content">
              <el-descriptions :column="4" border>
                <el-descriptions-item label="总行数">{{ reportData?.data_shape?.rows || 0 }}</el-descriptions-item>
                <el-descriptions-item label="总列数">{{ reportData?.data_shape?.columns || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失字段">{{ reportData?.quality_report?.missing?.length || 0 }}</el-descriptions-item>
                <el-descriptions-item label="重复记录">{{ reportData?.quality_report?.duplicates?.count || 0 }}</el-descriptions-item>
              </el-descriptions>
              <div class="type-distribution">
                <span class="label">变量类型分布：</span>
                <span v-for="(count, type) in typeCounts" :key="type" class="type-tag">
                  {{ typeDisplay[type] || type }}: {{ count }}
                </span>
              </div>
            </div>
          </el-collapse-item>

          <!-- 变量详情 -->
          <el-collapse-item name="variables">
            <template #title>
              <span>📋 变量详情</span>
            </template>
            <div class="detail-content">
              <el-table :data="variableList" border size="small" max-height="300">
                <el-table-column prop="name" label="变量名" />
                <el-table-column prop="type_desc" label="类型" />
                <el-table-column prop="count" label="样本量" />
                <el-table-column prop="missing" label="缺失数" />
                <el-table-column prop="missing_pct" label="缺失率">
                  <template #default="{ row }">
                    {{ row.missing_pct.toFixed(1) }}%
                  </template>
                </el-table-column>
                <el-table-column prop="center" label="中心趋势" />
                <el-table-column prop="spread" label="分布" />
              </el-table>
            </div>
          </el-collapse-item>

          <!-- 相关性分析 -->
          <el-collapse-item name="correlation">
            <template #title>
              <span>🔗 相关性分析</span>
            </template>
            <div class="detail-content">
              <div v-if="highCorrelations.length > 0">
                <p>发现 <strong>{{ highCorrelations.length }}</strong> 对强相关关系（|r| > 0.7）：</p>
                <el-table :data="highCorrelations" border size="small">
                  <el-table-column prop="var1" label="变量1" />
                  <el-table-column prop="var2" label="变量2" />
                  <el-table-column prop="value" label="相关系数" width="120">
                    <template #default="{ row }">
                      {{ row.value.toFixed(3) }}
                    </template>
                  </el-table-column>
                  <el-table-column label="方向" width="80">
                    <template #default="{ row }">
                      <el-tag :type="row.value > 0 ? 'danger' : 'success'" size="small">
                        {{ row.value > 0 ? '正相关' : '负相关' }}
                      </el-tag>
                    </template>
                  </el-table-column>
                </el-table>
              </div>
              <div v-else>
                <p>未发现强相关关系（|r| > 0.7）</p>
              </div>
            </div>
          </el-collapse-item>

          <!-- 时间序列分析 -->
          <el-collapse-item name="timeseries">
            <template #title>
              <span>📈 时间序列分析</span>
            </template>
            <div class="detail-content">
              <div v-if="timeSeriesData.length > 0">
                <el-table :data="timeSeriesData" border size="small">
                  <el-table-column prop="key" label="变量/分组" />
                  <el-table-column prop="n_samples" label="样本量" />
                  <el-table-column prop="stationary" label="平稳性" />
                  <el-table-column prop="autocorrelation" label="自相关性" />
                  <el-table-column prop="seasonality" label="季节性" />
                </el-table>
                <div v-if="hasAutoCorrelation" class="insight-hint">
                  ✅ 检测到自相关性，适合进行时间序列预测
                </div>
              </div>
              <div v-else>
                <p>未检测到时间序列数据</p>
              </div>
            </div>
          </el-collapse-item>

          <!-- 勾稽规则 -->
          <el-collapse-item name="audit">
            <template #title>
              <span>🔗 勾稽规则</span>
            </template>
            <div class="detail-content">
              <div v-if="auditRulesCount.arithmetic + auditRulesCount.functional + auditRulesCount.temporal > 0">
                <el-descriptions :column="3" border>
                  <el-descriptions-item label="数值关系">{{ auditRulesCount.arithmetic || 0 }}</el-descriptions-item>
                  <el-descriptions-item label="函数依赖">{{ auditRulesCount.functional || 0 }}</el-descriptions-item>
                  <el-descriptions-item label="时序约束">{{ auditRulesCount.temporal || 0 }}</el-descriptions-item>
                </el-descriptions>
                <div v-if="auditRulesList.length > 0" style="margin-top: 12px;">
                  <el-table :data="auditRulesList" border size="small">
                    <el-table-column prop="rule" label="规则" />
                    <el-table-column prop="confidence" label="置信度" width="100">
                      <template #default="{ row }">
                        {{ (row.confidence * 100).toFixed(1) }}%
                      </template>
                    </el-table-column>
                    <el-table-column prop="priority" label="优先级" width="80">
                      <template #default="{ row }">
                        <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
                          {{ row.priority }}
                        </el-tag>
                      </template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
              <div v-else>
                <p>未发现勾稽规则</p>
              </div>
            </div>
          </el-collapse-item>

          <!-- 清洗建议 -->
          <el-collapse-item name="cleaning">
            <template #title>
              <span>🧹 清洗建议</span>
            </template>
            <div class="detail-content">
              <ul v-if="cleaningSuggestions.length > 0">
                <li v-for="(s, index) in cleaningSuggestions" :key="index">{{ s }}</li>
              </ul>
              <p v-else>✅ 数据质量良好，无明显清洗需求</p>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>

      <!-- 操作按钮 -->
      <div class="actions">
        <el-button @click="goToQuality">⬅ 返回质量看板</el-button>
        <el-button @click="goToUpload">🔄 新建分析</el-button>
      </div>
    </div>

    <div v-else class="empty-state">
      <el-empty description="暂无报告数据，请先完成分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useAnalysisStore } from '../stores/analysis'
import { reportApi } from '../api/report'

const router = useRouter()
const route = useRoute()
const sessionStore = useSessionStore()
const analysisStore = useAnalysisStore()

const loading = ref(true)
const reportData = ref(null)
const summary = ref([])
const insightFindings = ref([])
const activeNames = ref(['overview'])

// 当前有效的 session_id
const currentSessionId = ref(null)

onMounted(async () => {
  await loadReport()
})

async function loadReport() {
  // 1. 从 sessionStore 获取
  let sessionId = sessionStore.currentSessionId

  // 2. 如果 sessionStore 没有，从 analysisStore 获取
  if (!sessionId && analysisStore.analysisSessionId) {
    sessionId = analysisStore.analysisSessionId
    console.log('从 analysisStore 获取 session_id:', sessionId)
  }

  // 3. 如果还是没有，从 URL 参数获取
  if (!sessionId) {
    sessionId = route.query.session
    console.log('从 URL 获取 session_id:', sessionId)
  }

  // 4. 如果还是没有，从浏览器 localStorage 获取（兜底）
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
    console.log('从 localStorage 获取 session_id:', sessionId)
  }

  if (!sessionId) {
    console.warn('没有 session_id，跳转到首页')
    router.push('/')
    return
  }

  currentSessionId.value = sessionId

  // 保存到 localStorage 供下次使用
  localStorage.setItem('lastSessionId', sessionId)

  // 确保 sessionStore 中有值
  if (!sessionStore.currentSessionId) {
    sessionStore.currentSessionId = sessionId
    try {
      await sessionStore.loadSession(sessionId)
    } catch (err) {
      console.warn('加载会话失败:', err)
    }
  }

  loading.value = true
  try {
    const [summaryResult, insightsResult, reportResult] = await Promise.all([
      reportApi.getSummary(sessionId),
      reportApi.getInsights(sessionId),
      reportApi.get(sessionId)
    ])

    summary.value = summaryResult.conclusions || []
    insightFindings.value = insightsResult.findings || []
    reportData.value = reportResult

    console.log('报告加载成功:', {
      summary: summary.value.length,
      insights: insightFindings.value.length,
      reportData: !!reportData.value
    })

  } catch (err) {
    console.error('加载报告失败:', err)
    ElMessage.error('加载报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

// ==================== 计算属性 ====================

const typeCounts = computed(() => {
  const variableTypes = reportData.value?.variable_types || {}
  const counts = {}
  for (const info of Object.values(variableTypes)) {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  }
  return counts
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

const variableList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type_desc: info.type_desc || info.type,
    count: info.count || 0,
    missing: info.missing || 0,
    missing_pct: info.missing_pct || 0,
    center: info.mean !== undefined ? info.mean.toFixed(2) : (info.mode || '-'),
    spread: info.std !== undefined ? `±${info.std.toFixed(2)}` : (info.n_unique ? `${info.n_unique}个类别` : '-')
  }))
})

const highCorrelations = computed(() => {
  return reportData.value?.correlations?.high_correlations || []
})

const timeSeriesData = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  return Object.entries(diag).map(([key, info]) => ({
    key,
    n_samples: info.n_samples || 0,
    stationary: info.is_stationary ? '✅ 平稳' : '⚠️ 非平稳',
    autocorrelation: info.has_autocorrelation ? '✅ 有' : '❌ 无',
    seasonality: info.has_seasonality ? '✅ 有' : '❌ 无'
  }))
})

const hasAutoCorrelation = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  return Object.values(diag).some(d => d.has_autocorrelation)
})

const auditRulesCount = computed(() => {
  const rules = reportData.value?.quality_report?.audit_rules || {}
  return {
    arithmetic: rules.arithmetic_rules?.length || 0,
    functional: rules.functional_dependencies?.length || 0,
    temporal: rules.temporal_rules?.length || 0
  }
})

const auditRulesList = computed(() => {
  const rules = reportData.value?.quality_report?.audit_rules || {}
  return [
    ...(rules.arithmetic_rules || []),
    ...(rules.functional_dependencies || []),
    ...(rules.temporal_rules || [])
  ]
})

const cleaningSuggestions = computed(() => {
  return reportData.value?.cleaning_suggestions || []
})

// ==================== 方法 ====================

function expandAll() {
  activeNames.value = ['overview', 'variables', 'correlation', 'timeseries', 'audit', 'cleaning']
}

function collapseAll() {
  activeNames.value = []
}

async function handleExport(format) {
  const sessionId = currentSessionId.value
  if (!sessionId) {
    ElMessage.warning('没有可导出的会话')
    return
  }

  try {
    const blob = await reportApi.export(sessionId, format)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report_${sessionId}.${format === 'html' ? 'html' : format === 'json' ? 'json' : 'xlsx'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    ElMessage.success(`导出成功: ${format.toUpperCase()}`)
  } catch (err) {
    ElMessage.error('导出失败: ' + err.message)
  }
}

// 在 <script setup> 中添加方法
async function handleExportLog() {
  const sessionId = currentSessionId.value
  if (!sessionId) {
    ElMessage.warning('没有可导出的会话')
    return
  }
  try {
    const blob = await reportApi.exportLog(sessionId)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `analysis_log_${sessionId}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    ElMessage.success('日志导出成功')
  } catch (err) {
    ElMessage.error('日志导出失败: ' + err.message)
  }
}

function goToQuality() {
  router.push('/quality')
}

function goToUpload() {
  router.push('/upload')
}
</script>

<style scoped>
.report-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
}
.report-header h2 {
  margin: 0;
}
.report-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.loading-container {
  padding: 40px 0;
}
.empty-state {
  padding: 60px 0;
}
.report-content > div {
  margin-bottom: 30px;
}
.report-content h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}

.conclusion-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}
.conclusion-card {
  text-align: center;
  padding: 16px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.conclusion-icon {
  font-size: 24px;
  margin-bottom: 8px;
}
.conclusion-title {
  font-weight: bold;
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 4px;
}
.conclusion-desc {
  font-size: 12px;
  color: #909399;
  line-height: 1.4;
}

.detail-content {
  padding: 10px 0;
}
.detail-content p {
  color: #666;
}
.type-distribution {
  margin-top: 12px;
  font-size: 13px;
}
.type-distribution .label {
  color: #666;
}
.type-distribution .type-tag {
  display: inline-block;
  margin: 0 8px 4px 0;
  background: #f0f2f6;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
}
.insight-hint {
  margin-top: 12px;
  padding: 8px 12px;
  background: #e8f5e9;
  border-radius: 6px;
  color: #2e7d32;
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>