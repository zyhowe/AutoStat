<template>
  <div class="quality-page">
    <h2>📊 数据质量看板</h2>
    <p class="subtitle">四维质量评分 + SLA 合规性 + 缺失共现分析，全面了解数据健康状况</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="quality-content">
      <!-- ===== 表选择器 ===== -->
      <TableSelector
        v-model="currentTable"
        :table-names="tableNames"
        :is-multi-table="isMultiTable"
        @change="onTableChange"
      />

      <!-- ===== SLA 红绿灯（新增） ===== -->
      <div class="sla-section">
        <div class="sla-title">🚦 SLA 合规性</div>
        <div class="sla-grid">
          <div class="sla-item" :class="slaStatus.completeness ? 'pass' : 'fail'">
            <span class="sla-label">完整性 ≤20% 缺失</span>
            <span class="sla-value">{{ slaStatus.completeness ? '✅' : '❌' }}</span>
            <span class="sla-detail">{{ slaStatus.completeness ? '达标' : '不达标' }}</span>
          </div>
          <div class="sla-item" :class="slaStatus.accuracy ? 'pass' : 'fail'">
            <span class="sla-label">准确性 ≤5% 异常</span>
            <span class="sla-value">{{ slaStatus.accuracy ? '✅' : '❌' }}</span>
            <span class="sla-detail">{{ slaStatus.accuracy ? '达标' : '不达标' }}</span>
          </div>
          <div class="sla-item" :class="slaStatus.noEmptyColumns ? 'pass' : 'fail'">
            <span class="sla-label">无全空列</span>
            <span class="sla-value">{{ slaStatus.noEmptyColumns ? '✅' : '❌' }}</span>
            <span class="sla-detail">{{ slaStatus.noEmptyColumns ? '达标' : '不达标' }}</span>
          </div>
          <div class="sla-item" :class="slaStatus.unitConsistent ? 'pass' : 'fail'">
            <span class="sla-label">单位一致</span>
            <span class="sla-value">{{ slaStatus.unitConsistent ? '✅' : '❌' }}</span>
            <span class="sla-detail">{{ slaStatus.unitConsistent ? '达标' : '不达标' }}</span>
          </div>
          <div class="sla-item" :class="slaStatus.namingReadable ? 'pass' : 'fail'">
            <span class="sla-label">字段命名可读</span>
            <span class="sla-value">{{ slaStatus.namingReadable ? '✅' : '❌' }}</span>
            <span class="sla-detail">{{ slaStatus.namingReadable ? '达标' : '不达标' }}</span>
          </div>
        </div>
        <div class="sla-summary">
          达标率：{{ slaPassCount }} / 5
          <el-progress :percentage="slaPassCount / 5 * 100" :color="slaPassCount >= 4 ? '#67c23a' : slaPassCount >= 3 ? '#e6a23c' : '#f56c6c'" :stroke-width="8" />
        </div>
      </div>

      <!-- 图表区域 -->
      <div class="charts-row">
        <div class="chart-card gauge-card">
          <div class="chart-header">
            <span class="chart-title">🎯 综合评分</span>
          </div>
          <v-chart
            v-if="hasGaugeData"
            :key="'gauge_' + gaugeKey"
            :option="gaugeOption"
            class="chart-container"
          />
          <div v-else class="chart-empty">暂无评分数据</div>
        </div>
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 四维质量得分</span>
          </div>
          <v-chart v-if="hasRadarData" :option="radarOption" class="chart-container" />
          <div v-else class="chart-empty">暂无维度数据</div>
        </div>
      </div>

      <!-- 四维评分卡片 -->
      <div class="dimensions">
        <el-card
          v-for="(score, name) in filteredDimensions"
          :key="name"
          class="dimension-card"
          shadow="hover"
        >
          <div class="dimension-header">
            <span class="dimension-name">{{ getDimensionLabel(name) }}</span>
            <el-tooltip
              placement="top"
              effect="dark"
              :content="getDimensionTooltip(name)"
            >
              <el-icon class="dimension-help"><QuestionFilled /></el-icon>
            </el-tooltip>
          </div>
          <el-progress
            :percentage="Math.round(score)"
            :color="getProgressColor(score)"
            :stroke-width="12"
          />
          <div class="dimension-value">{{ Math.round(score) }}%</div>
        </el-card>
      </div>

      <!-- ===== 缺失共现簇（新增） ===== -->
      <div v-if="missingClusters.length > 0" class="cluster-section">
        <h3>📊 缺失共现簇</h3>
        <p class="cluster-desc">这些字段的缺失模式高度同步，可能来自同一数据源或子表</p>
        <div class="cluster-grid">
          <div v-for="(cluster, idx) in missingClusters" :key="idx" class="cluster-card">
            <div class="cluster-header">
              <span class="cluster-id">簇 {{ idx + 1 }}</span>
              <span class="cluster-size">{{ cluster.fields.length }} 个字段</span>
            </div>
            <div class="cluster-fields">
              <el-tag v-for="field in cluster.fields" :key="field" size="small" type="info">{{ field }}</el-tag>
            </div>
            <div class="cluster-missing-rate">缺失率：{{ cluster.avgMissingRate }}%</div>
          </div>
        </div>
      </div>

      <!-- 问题清单 -->
      <div class="issues-section">
        <h3>⚠️ 问题清单</h3>
        <el-table :data="issues" border style="width: 100%" max-height="400">
          <el-table-column prop="level" label="级别" width="100" align="center">
            <template #default="{ row }">
              <el-tag :type="row.level === 'error' ? 'danger' : 'warning'" size="small">
                {{ row.level === 'error' ? '严重' : '警告' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="field" label="字段" width="150">
            <template #default="{ row }">
              <span class="field-name-link" @click="openFieldDetail(row.field)">{{ row.field }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="message" label="问题描述" min-width="200">
            <template #default="{ row }">
              <span class="field-name-link" @click="showDataByAlert(row)">
                {{ row.message }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="current" label="当前值" width="120" align="center" />
          <el-table-column prop="threshold" label="阈值" width="120" align="center" />
        </el-table>
        <div v-if="issues.length === 0" class="empty-text">✅ 未发现问题</div>
      </div>
    </div>

    <div v-else-if="!loading" class="empty-state">
      <el-empty description="暂无质量数据，请先完成分析">
        <el-button type="primary" @click="goTo('upload')">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { QuestionFilled } from '@element-plus/icons-vue'
import { useSessionStore } from '../stores/session'
import { useFieldDetailStore } from '../stores/fieldDetail'
import { reportApi } from '../api/report'
import { openDataPreview } from '../components/DataPreviewDialog'
import TableSelector from '../components/TableSelector.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const fieldDetailStore = useFieldDetailStore()

const loading = ref(true)
const reportData = ref(null)
const issues = ref([])
const gaugeKey = ref(0)

// ===== 表选择器状态 =====
const currentTable = ref('merged')
const tableNames = ref([])
const isMultiTable = ref(false)

const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

// ===== 当前数据 =====
const currentData = computed(() => {
  if (!reportData.value?.all_tables) return {}
  return reportData.value.all_tables[currentTable.value] || reportData.value.all_tables['merged'] || {}
})

const currentQualityData = computed(() => {
  return currentData.value?.quality_report || {}
})

const currentDimensions = computed(() => {
  return currentQualityData.value?.dimensions || {}
})

const currentOverallScore = computed(() => {
  return currentQualityData.value?.overall_score ?? null
})

const currentAlerts = computed(() => {
  return currentQualityData.value?.alerts || []
})

// ===== SLA 状态（新增） =====
const slaStatus = computed(() => {
  const missingList = currentQualityData.value?.missing || []
  const outliers = currentQualityData.value?.outliers || {}
  const columnNames = currentData.value?.column_names || []

  // 完整性：缺失率 ≤20% 的列占比
  const completenessPass = missingList.filter(m => m.percent > 20).length === 0

  // 准确性：异常率 ≤5% 的列占比
  const accuracyPass = Object.values(outliers).every(o => o.percent <= 5)

  // 无全空列
  const noEmptyColumns = !missingList.some(m => m.percent === 100)

  // 单位一致：简单检查 cunit 是否只有一种单位
  const unitInfo = currentData.value?.variable_summaries?.cunit
  const unitConsistent = unitInfo ? unitInfo.n_unique === 1 : true

  // 字段命名可读：数字后缀占比 < 30%
  const namingReadable = columnNames.filter(col => /^companyfixasset\d+$/.test(col)).length / (columnNames.length || 1) < 0.3

  return {
    completeness: completenessPass,
    accuracy: accuracyPass,
    noEmptyColumns,
    unitConsistent,
    namingReadable
  }
})

const slaPassCount = computed(() => {
  return Object.values(slaStatus.value).filter(v => v).length
})

// ===== 缺失共现簇（新增） =====
const missingClusters = computed(() => {
  const missingList = currentQualityData.value?.missing || []
  if (missingList.length < 2) return []

  // 按缺失率分组，找出缺失率相近且字段名相似的簇
  const clusters = []
  const grouped = {}
  for (const item of missingList) {
    const pct = Math.round(item.percent / 10) * 10 // 按 10% 分桶
    if (!grouped[pct]) grouped[pct] = []
    grouped[pct].push(item.column)
  }

  // 只保留缺失率 > 50% 的簇，且字段数 >= 2
  for (const [pct, fields] of Object.entries(grouped)) {
    if (parseInt(pct) > 50 && fields.length >= 2) {
      // 检查字段名是否相似（如都有 companyfixasset 前缀）
      const prefixFields = fields.filter(f => f.startsWith('companyfixasset'))
      if (prefixFields.length >= 2) {
        clusters.push({
          fields: prefixFields,
          avgMissingRate: parseInt(pct)
        })
      }
    }
  }

  // 按字段数降序
  clusters.sort((a, b) => b.fields.length - a.fields.length)
  return clusters.slice(0, 5)
})

// ===== 原有逻辑 =====
watch(() => currentOverallScore.value, (newVal) => {
  if (newVal !== undefined && newVal !== null) {
    gaugeKey.value += 1
  }
})

function onTableChange() {
  gaugeKey.value += 1
  updateIssues()
}

onMounted(async () => {
  await loadQuality()
})

async function loadQuality() {
  let sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
  }
  if (!sessionId) {
    router.push('/')
    return
  }
  if (!sessionStore.currentSessionId) {
    sessionStore.currentSessionId = sessionId
  }

  loading.value = true
  try {
    const reportResult = await reportApi.get(sessionId)
    reportData.value = reportResult

    const allTables = reportResult?.all_tables || {}
    const tableKeys = Object.keys(allTables)
    tableNames.value = tableKeys.filter(k => k !== 'merged')
    isMultiTable.value = tableNames.value.length > 1

    if (!currentTable.value || !allTables[currentTable.value]) {
      currentTable.value = 'merged'
    }

    updateIssues()

  } catch (err) {
    ElMessage.error('加载质量报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function updateIssues() {
  const allIssues = []
  const alerts = currentAlerts.value || []
  alerts.forEach(alert => {
    if (alert.level === 'error' || alert.level === 'warning') {
      allIssues.push({
        level: alert.level,
        field: alert.field || alert.dimension,
        message: alert.message,
        current: alert.current,
        threshold: alert.threshold
      })
    }
  })
  issues.value = allIssues
}

function getDimensionLabel(name) {
  const labels = {
    'completeness': '完整性',
    'accuracy': '准确性',
    'consistency': '一致性',
    'uniqueness': '唯一性'
  }
  return labels[name] || name
}

function getDimensionTooltip(name) {
  const tooltips = {
    'completeness': '数据是否完整，有没有空值\n计算方式：非空值数量 / 总行数 × 100%',
    'accuracy': '数据是否正确，有没有异常值\n计算方式：1 - (异常值数量 / 有效样本量) × 100%',
    'consistency': '数据是否统一，勾稽关系是否成立\n计算方式：满足勾稽规则的记录数 / 总记录数 × 100%',
    'uniqueness': '数据是否重复\n计算方式：1 - (重复记录数 / 总行数) × 100%'
  }
  return tooltips[name] || '该维度评分'
}

function getProgressColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}

const filteredDimensions = computed(() => {
  const dims = currentDimensions.value || {}
  const filtered = {}
  Object.keys(dims).forEach(k => {
    if (k !== 'timeliness') {
      filtered[k] = dims[k]
    }
  })
  return filtered
})

const hasGaugeData = computed(() => {
  const score = currentOverallScore.value
  return score !== undefined && score !== null && !isNaN(Number(score))
})

const gaugeOption = computed(() => {
  const rawScore = currentOverallScore.value
  const score = Number(rawScore) || 0
  const color = score >= 80 ? '#67C23A' : score >= 60 ? '#E6A23C' : '#F56C6C'
  return {
    series: [{
      type: 'gauge',
      center: ['50%', '55%'],
      radius: '85%',
      startAngle: 210,
      endAngle: -30,
      min: 0,
      max: 100,
      splitNumber: 5,
      progress: {
        show: true,
        width: 14,
        roundCap: true,
        itemStyle: { color }
      },
      axisLine: {
        lineStyle: {
          width: 14,
          color: [
            [0.3, '#F56C6C'],
            [0.7, '#E6A23C'],
            [1, '#67C23A']
          ]
        }
      },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      pointer: { show: false },
      anchor: { show: false },
      title: { show: false },
      detail: {
        valueAnimation: true,
        formatter: function(params) {
          const val = typeof params === 'object' ? (params.value || 0) : (params || 0)
          return Number(val).toFixed(1) + ' 分'
        },
        color: '#2C3E50',
        fontSize: 24,
        fontWeight: 'bold',
        offsetCenter: [0, '30%']
      },
      data: [{ value: score }]
    }]
  }
})

const hasRadarData = computed(() => {
  return Object.keys(filteredDimensions.value).length > 0
})

const radarOption = computed(() => {
  const dims = filteredDimensions.value
  const labels = {
    completeness: '完整性',
    accuracy: '准确性',
    consistency: '一致性',
    uniqueness: '唯一性'
  }
  const indicator = Object.keys(dims).map(key => ({
    name: labels[key] || key,
    max: 100
  }))
  const values = Object.values(dims).map(v => Math.round(v))
  return {
    tooltip: { trigger: 'item' },
    legend: { show: false },
    radar: {
      indicator,
      shape: 'circle',
      center: ['50%', '50%'],
      radius: '70%',
      axisName: { color: '#333', fontSize: 12 },
      splitArea: { areaStyle: { color: ['rgba(64,158,255,0.02)'] } }
    },
    series: [{
      type: 'radar',
      data: [{ value: values, name: '质量得分' }],
      areaStyle: { color: 'rgba(64,158,255,0.3)' },
      lineStyle: { color: '#409EFF', width: 2 },
      itemStyle: { color: '#409EFF' }
    }]
  }
})

// ===== 字段详情 =====
function buildFieldData(fieldName) {
  const summaries = currentData.value?.variable_summaries || {}
  const variableTypes = currentData.value?.variable_types || {}
  const qualityReport = currentQualityData.value

  const summary = summaries?.[fieldName] || {}
  const varType = variableTypes?.[fieldName]?.type || 'unknown'
  const varTypeDesc = variableTypes?.[fieldName]?.type_desc || typeDisplay[varType] || varType

  const tsDiag = currentData.value?.time_series_diagnostics?.[fieldName] || null
  const outlier = qualityReport?.outliers?.[fieldName] || null
  const missing = qualityReport?.missing?.find(m => m.column === fieldName) || null
  const dupInfo = qualityReport?.duplicates || null

  const correlations = []
  const matrix = currentData.value?.correlations?.matrix || {}
  if (matrix[fieldName]) {
    const entries = Object.entries(matrix[fieldName])
    for (const [varName, value] of entries) {
      if (varName !== fieldName && value !== null && value !== undefined && Math.abs(value) >= 0.7) {
        correlations.push({ var: varName, value: parseFloat(Number(value).toFixed(4)) })
      }
    }
    correlations.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  }

  const rules = []
  const auditRules = qualityReport?.audit_rules || {}
  const allRules = [
    ...(auditRules.arithmetic_rules || []),
    ...(auditRules.functional_dependencies || []),
    ...(auditRules.temporal_rules || [])
  ]
  for (const rule of allRules) {
    if (rule.fields && rule.fields.includes(fieldName)) {
      rules.push({
        type: rule.relation_type === 'additive' ? '数值' :
              rule.rule?.includes('→') ? '函数依赖' : '时序约束',
        rule: rule.rule,
        confidence: rule.confidence || 1.0
      })
    }
  }

  const models = []
  const modelRecs = currentData.value?.model_recommendations || []
  for (const rec of modelRecs) {
    let role = ''
    if (rec.target_column === fieldName) {
      role = '🎯 目标'
    } else if (rec.feature_columns && rec.feature_columns.includes(fieldName)) {
      role = '📊 特征'
    }
    if (role) {
      models.push({
        role: role,
        task_type: rec.task_type || '',
        model: rec.ml || rec.model || '',
        target: rec.target_column || ''
      })
    }
  }

  let topCategories = []
  if (summary.top_categories && Object.keys(summary.top_categories).length > 0) {
    const entries = Object.entries(summary.top_categories)
    const total = entries.reduce((s, e) => s + e[1], 0)
    topCategories = entries.map(([name, count]) => ({
      name: String(name),
      count: count,
      pct: total > 0 ? (count / total * 100) : 0
    })).sort((a, b) => b.count - a.count)
  } else if (summary.value_counts && Object.keys(summary.value_counts).length > 0) {
    const entries = Object.entries(summary.value_counts)
    const total = entries.reduce((s, e) => s + e[1], 0)
    topCategories = entries.map(([name, count]) => ({
      name: String(name),
      count: count,
      pct: total > 0 ? (count / total * 100) : 0
    })).sort((a, b) => b.count - a.count)
  }

  return {
    fieldName,
    varType,
    varTypeDesc,
    summary: { ...summary, topCategories },
    tsDiag,
    outlier,
    missing,
    duplicateInfo: dupInfo,
    correlations,
    rules,
    models,
    topCategories
  }
}

function openFieldDetail(fieldName) {
  const data = buildFieldData(fieldName)
  fieldDetailStore.open(fieldName, data)
}

// ==================== 数据预览联动 ====================
function showDataByAlert(alert) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  const field = alert.field || alert.dimension
  const message = alert.message || ''

  let filters = []
  let title = '数据详情'

  if (field) {
    if (message.includes('缺失')) {
      filters = [{ field: field, condition: 'is_null', value: true }]
      title = `「${field}」缺失的数据`
    } else if (message.includes('异常')) {
      filters = [{ field: field, condition: 'is_outlier', value: true }]
      title = `「${field}」异常值数据`
    } else {
      filters = []
      title = `「${field}」所有数据`
    }
  } else {
    ElMessage.warning('无法解析该告警的筛选条件')
    return
  }

  openDataPreview({
    sessionId: sessionId,
    title: title,
    filters: filters
  })
}

function goTo(routeName) {
  router.push(`/${routeName}`)
}

// ===== 监听表切换 =====
watch(currentTable, () => {
  gaugeKey.value += 1
  updateIssues()
})
</script>

<style scoped>
.quality-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px 20px 80px 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}
.loading-container {
  padding: 40px 0;
}
.empty-state {
  padding: 60px 0;
}

/* ===== SLA 样式 ===== */
.sla-section {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  padding: 16px 20px;
  margin-bottom: 24px;
}
.sla-title {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 12px;
}
.sla-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.sla-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  background: #f5f7fa;
}
.sla-item.pass {
  border-left: 4px solid #67c23a;
}
.sla-item.fail {
  border-left: 4px solid #f56c6c;
}
.sla-label {
  font-size: 12px;
  color: #555;
}
.sla-value {
  font-size: 16px;
  margin-left: auto;
}
.sla-detail {
  font-size: 11px;
  color: #909399;
}
.sla-summary {
  display: flex;
  align-items: center;
  gap: 16px;
  padding-top: 8px;
  border-top: 1px solid #e8ecf1;
  font-size: 13px;
  color: #2c3e50;
}
.sla-summary .el-progress {
  flex: 1;
}

/* ===== 缺失共现簇 ===== */
.cluster-section {
  margin-bottom: 30px;
}
.cluster-section h3 {
  margin-bottom: 4px;
  color: #2c3e50;
}
.cluster-desc {
  color: #909399;
  font-size: 13px;
  margin-bottom: 12px;
}
.cluster-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
.cluster-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 12px 16px;
  border: 1px solid #e4e7ed;
}
.cluster-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.cluster-id {
  font-weight: 600;
  font-size: 13px;
  color: #2c3e50;
}
.cluster-size {
  font-size: 11px;
  color: #909399;
}
.cluster-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}
.cluster-missing-rate {
  font-size: 11px;
  color: #e6a23c;
}

/* ===== 原有样式 ===== */
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}
.chart-card {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  padding: 16px 16px 4px 16px;
  transition: box-shadow 0.2s;
}
.chart-card:hover {
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.chart-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}
.chart-container {
  width: 100%;
  height: 200px;
}
.chart-empty {
  height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #bbb;
  font-size: 13px;
}
.gauge-card .chart-container {
  height: 180px;
}

@media (max-width: 768px) {
  .charts-row {
    grid-template-columns: 1fr;
  }
  .chart-container {
    height: 180px;
  }
}

.dimensions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 30px;
}
.dimension-card {
  padding: 16px;
}
.dimension-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.dimension-name {
  font-weight: 500;
  font-size: 15px;
  color: #2c3e50;
}
.dimension-help {
  font-size: 16px;
  color: #909399;
  cursor: pointer;
}
.dimension-help:hover {
  color: #409EFF;
}
.dimension-value {
  text-align: right;
  margin-top: 4px;
  font-weight: 500;
  font-size: 14px;
  color: #666;
}

.issues-section {
  margin-bottom: 30px;
}
.issues-section h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-text {
  padding: 20px;
  text-align: center;
  color: #67c23a;
}

.field-name-link {
  color: #409EFF;
  cursor: pointer;
  font-weight: 500;
  transition: color 0.2s;
}
.field-name-link:hover {
  color: #66b1ff;
  text-decoration: underline;
}
</style>