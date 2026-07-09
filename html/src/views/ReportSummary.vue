<template>
  <div class="report-summary">
    <div class="report-header">
      <h2>📊 分析总览</h2>
      <div class="header-actions">
        <el-button size="small" type="primary" plain @click="handleDownloadHtml">
          📄 导出HTML
        </el-button>
        <el-button size="small" type="success" plain @click="handleExport('json')">
          📋 导出JSON
        </el-button>
      </div>
    </div>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="summary-content">
      <!-- ==================== 关键指标卡片 ==================== -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.rows || 0 }}</div>
          <div class="stat-label">总行数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.columns || 0 }}</div>
          <div class="stat-label">总列数</div>
        </div>
        <div class="stat-card" :class="outlierClass">
          <div class="stat-value">{{ outlierCount }}</div>
          <div class="stat-label">异常字段</div>
        </div>
        <div class="stat-card" :class="highMissingClass">
          <div class="stat-value">{{ highMissingCount }}</div>
          <div class="stat-label">高缺失字段</div>
        </div>
        <div class="stat-card" :class="ruleClass">
          <div class="stat-value">{{ auditRulesTotal }}</div>
          <div class="stat-label">勾稽规则</div>
        </div>
        <div class="stat-card" :class="corrCountClass">
          <div class="stat-value">{{ highCorrCount }}</div>
          <div class="stat-label">强相关对数</div>
        </div>
        <div class="stat-card" :class="modelCountClass">
          <div class="stat-value">{{ modelRecommendCount }}</div>
          <div class="stat-label">可预测模型</div>
        </div>
      </div>

      <!-- ==================== 图表区域 ==================== -->
      <div class="charts-row">
        <!-- 雷达图 -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 质量维度得分</span>
          </div>
          <v-chart v-if="hasRadarData" :option="radarOption" class="chart-container" />
          <div v-else class="chart-empty">暂无质量维度数据</div>
        </div>

        <!-- 仪表盘 -->
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
      </div>

      <!-- ==================== 核心发现 ==================== -->
      <div class="section">
        <h3>📊 核心发现</h3>
        <div v-if="!hasDiscoveries" class="empty-tip">
          💡 数据分布较为均匀，未发现明显的极端规律或数据质量问题
        </div>

        <div v-else class="discovery-groups">
          <!-- 数据概况 -->
          <div class="discovery-group field-group">
            <div class="group-header">
              <span class="group-icon">📋</span>
              <span class="group-title">数据概况</span>
            </div>
            <div class="group-content">
              <div class="field-item">
                <span class="field-label">总字段数：</span>
                <span class="field-value">{{ reportData.data_shape?.columns || 0 }}</span>
              </div>
              <div class="field-item">
                <span class="field-label">连续变量：</span>
                <span class="field-value">{{ numericCount }}</span>
              </div>
              <div class="field-item">
                <span class="field-label">分类变量：</span>
                <span class="field-value">{{ categoricalCount }}</span>
              </div>
              <div class="field-item">
                <span class="field-label">日期变量：</span>
                <span class="field-value">{{ datetimeCount }}</span>
              </div>
              <div class="field-item">
                <span class="field-label">样本量：</span>
                <span class="field-value">{{ reportData.data_shape?.rows || 0 }}</span>
              </div>
            </div>
          </div>

          <!-- 质量发现 -->
          <div v-if="qualityDiscoveries.length > 0" class="discovery-group quality-group">
            <div class="group-header">
              <span class="group-icon">🔍</span>
              <span class="group-title">质量诊断</span>
            </div>
            <ul class="discovery-list">
              <li v-for="(item, index) in qualityDiscoveries" :key="'quality_' + index">
                {{ item }}
              </li>
            </ul>
          </div>

          <!-- 规律发现 -->
          <div v-if="patternDiscoveries.length > 0" class="discovery-group pattern-group">
            <div class="group-header">
              <span class="group-icon">📈</span>
              <span class="group-title">关联规律</span>
            </div>
            <ul class="discovery-list">
              <li v-for="(item, index) in patternDiscoveries" :key="'pattern_' + index">
                {{ item }}
              </li>
            </ul>
          </div>

          <!-- 建模建议 -->
          <div v-if="predictionDiscoveries.length > 0" class="discovery-group prediction-group">
            <div class="group-header">
              <span class="group-icon">🤖</span>
              <span class="group-title">建模建议</span>
            </div>
            <ul class="discovery-list">
              <li v-for="(item, index) in predictionDiscoveries" :key="'pred_' + index">
                {{ item }}
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="!loading" class="empty-state">
      <el-empty description="暂无数据，请先上传并完成分析">
        <el-button type="primary" @click="goTo('upload')">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)
const qualityData = ref(null)

const gaugeKey = ref(0)

watch(() => qualityData.value?.overall_score, (newVal) => {
  if (newVal !== undefined && newVal !== null) {
    nextTick(() => { gaugeKey.value += 1 })
  }
})

// ==================== 关键指标卡片 ====================
const highCorrCount = computed(() => {
  return reportData.value?.correlations?.high_correlations?.length || 0
})

const corrCountClass = computed(() => {
  const count = highCorrCount.value
  if (count === 0) return 'status-ok'
  if (count <= 10) return 'status-warn'
  return 'status-high'
})

const modelRecommendCount = computed(() => {
  return reportData.value?.model_recommendations?.length || 0
})

const modelCountClass = computed(() => {
  const count = modelRecommendCount.value
  if (count === 0) return 'status-ok'
  if (count <= 10) return 'status-warn'
  return 'status-high'
})

const outlierCount = computed(() => {
  const outliers = qualityData.value?.outliers || {}
  return Object.keys(outliers).length
})

const outlierClass = computed(() => {
  const count = outlierCount.value
  if (count === 0) return 'status-ok'
  if (count <= 3) return 'status-warn'
  return 'status-bad'
})

const highMissingCount = computed(() => {
  const missing = reportData.value?.quality_report?.missing || []
  return missing.filter(m => parseFloat(m.percent) > 20).length
})

const highMissingClass = computed(() => {
  const count = highMissingCount.value
  if (count === 0) return 'status-ok'
  if (count <= 3) return 'status-warn'
  return 'status-bad'
})

const auditRulesTotal = computed(() => {
  const rules = reportData.value?.quality_report?.audit_rules || {}
  return (rules.arithmetic_rules?.length || 0) +
         (rules.functional_dependencies?.length || 0) +
         (rules.temporal_rules?.length || 0)
})

const ruleClass = computed(() => {
  const count = auditRulesTotal.value
  if (count === 0) return 'status-ok'
  if (count <= 10) return 'status-warn'
  return 'status-bad'
})

// ==================== 数据概况统计 ====================
const variableTypes = computed(() => {
  return reportData.value?.variable_types || {}
})

const numericCount = computed(() => {
  return Object.values(variableTypes.value).filter(v => v.type === 'continuous').length
})

const categoricalCount = computed(() => {
  return Object.values(variableTypes.value).filter(v =>
    ['categorical', 'categorical_numeric', 'ordinal'].includes(v.type)
  ).length
})

const datetimeCount = computed(() => {
  return Object.values(variableTypes.value).filter(v => v.type === 'datetime').length
})

// ==================== 雷达图 ====================
const hasRadarData = computed(() => {
  const dims = qualityData.value?.dimensions || {}
  const filteredKeys = Object.keys(dims).filter(k => k !== 'timeliness')
  return filteredKeys.length > 0
})

const radarOption = computed(() => {
  const dims = qualityData.value?.dimensions || {}
  const filteredDims = {}
  Object.keys(dims).forEach(k => {
    if (k !== 'timeliness') {
      filteredDims[k] = dims[k]
    }
  })
  const labels = {
    completeness: '完整性',
    accuracy: '准确性',
    consistency: '一致性',
    uniqueness: '唯一性'
  }
  const indicator = Object.keys(filteredDims).map(key => ({
    name: labels[key] || key,
    max: 100
  }))
  const values = Object.values(filteredDims).map(v => Math.round(v))
  return {
    tooltip: { trigger: 'item' },
    legend: { show: false },
    radar: {
      indicator: indicator,
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

// ==================== 仪表盘 ====================
const hasGaugeData = computed(() => {
  const score = qualityData.value?.overall_score
  return score !== undefined && score !== null && !isNaN(Number(score))
})

const gaugeOption = computed(() => {
  const rawScore = qualityData.value?.overall_score
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
        itemStyle: { color: color }
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

// ==================== 核心发现 ====================
const qualityDiscoveries = computed(() => {
  const result = []
  const quality = reportData.value?.quality_report || {}
  const missing = quality.missing || []
  const outliers = quality.outliers || {}
  const duplicates = quality.duplicates || {}

  if (qualityData.value?.overall_score !== undefined && qualityData.value?.overall_score !== null) {
    const score = qualityData.value.overall_score
    const grade = score >= 80 ? '良好' : score >= 70 ? '一般' : '需关注'
    result.push(`综合质量评分 ${score} 分（${grade}）`)
  }

  const highMissing = missing.filter(m => parseFloat(m.percent) > 20)
  if (highMissing.length > 0) {
    const fields = highMissing.slice(0, 3).map(m => m.column).filter(Boolean)
    let text = `发现${highMissing.length}个字段缺失率超过20%`
    if (fields.length > 0) {
      text += `（${fields.join('、')}`
      if (highMissing.length > 3) text += `等${highMissing.length}个`
      text += '）'
    }
    text += '，建议填充或删除'
    result.push(text)
  }

  const outlierFields = Object.keys(outliers)
  if (outlierFields.length > 0) {
    const fields = outlierFields.slice(0, 3)
    let text = `发现${outlierFields.length}个字段存在异常值`
    if (fields.length > 0) {
      text += `（${fields.join('、')}`
      if (outlierFields.length > 3) text += `等${outlierFields.length}个`
      text += '）'
    }
    text += '，建议检查数据来源'
    result.push(text)
  }

  const dupCount = parseInt(duplicates.count) || 0
  if (dupCount > 0) {
    result.push(`发现${dupCount}条重复记录，建议去重处理`)
  }

  const rules = reportData.value?.quality_report?.audit_rules || {}
  const arithmeticCount = rules.arithmetic_rules?.length || 0
  const temporalCount = rules.temporal_rules?.length || 0
  const functionalCount = rules.functional_dependencies?.length || 0
  const totalRules = arithmeticCount + temporalCount + functionalCount
  if (totalRules > 0) {
    let detail = []
    if (arithmeticCount > 0) detail.push(`数值关系${arithmeticCount}条`)
    if (temporalCount > 0) detail.push(`时序约束${temporalCount}条`)
    if (functionalCount > 0) detail.push(`函数依赖${functionalCount}条`)
    result.push(`发现${totalRules}条勾稽规则（${detail.join('，')}）`)
  }

  return result
})

const patternDiscoveries = computed(() => {
  const result = []
  const correlations = reportData.value?.correlations || {}
  const tsDiag = reportData.value?.time_series_diagnostics || {}
  const distribution = reportData.value?.distribution_insights || {}

  const highCorrs = correlations.high_correlations || []
  if (highCorrs.length > 0) {
    const pairs = highCorrs.slice(0, 3).map(c => `${c.var1} ↔ ${c.var2} (r=${c.value})`)
    let text = `发现${highCorrs.length}对强相关关系`
    if (pairs.length > 0) {
      text += `（${pairs.join('、')}`
      if (highCorrs.length > 3) text += `等${highCorrs.length}对`
      text += '）'
    }
    text += '，建议重点关注'
    result.push(text)
  }

  const tsVars = Object.keys(tsDiag).filter(k => {
    const val = tsDiag[k]?.has_autocorrelation
    if (typeof val === 'string') return val.toLowerCase() === 'true'
    return val === true
  })
  if (tsVars.length > 0) {
    const fields = tsVars.slice(0, 3)
    let text = `检测到${tsVars.length}个序列存在自相关性`
    if (fields.length > 0) {
      text += `（${fields.join('、')}`
      if (tsVars.length > 3) text += `等${tsVars.length}个`
      text += '）'
    }
    text += '，适合进行时间序列预测'
    result.push(text)
  }

  const skewed = distribution.skewed_variables || []
  if (skewed.length > 0) {
    const names = skewed.slice(0, 3).map(s => s.name).filter(Boolean)
    let text = `发现${skewed.length}个偏态变量`
    if (names.length > 0) {
      text += `（${names.join('、')}`
      if (skewed.length > 3) text += `等${skewed.length}个`
      text += '）'
    }
    text += '，建议使用中位数描述'
    result.push(text)
  }

  const imbalanced = distribution.imbalanced_categoricals || []
  if (imbalanced.length > 0) {
    const names = imbalanced.slice(0, 3).map(s => s.name).filter(Boolean)
    let text = `发现${imbalanced.length}个不平衡分类变量`
    if (names.length > 0) {
      text += `（${names.join('、')}`
      if (imbalanced.length > 3) text += `等${imbalanced.length}个`
      text += '）'
    }
    text += '，分析时需注意类别失衡'
    result.push(text)
  }

  return result
})

const predictionDiscoveries = computed(() => {
  const result = []
  const variableTypesData = reportData.value?.variable_types || {}
  const dataShape = reportData.value?.data_shape || {}
  const modelRecs = reportData.value?.model_recommendations || []

  const targets = []
  for (const rec of modelRecs) {
    const target = rec.target_column
    if (target && !targets.includes(target)) {
      targets.push(target)
    }
  }
  if (targets.length > 0) {
    const fields = targets.slice(0, 3)
    let text = `${targets.length}个字段可预测`
    if (fields.length > 0) {
      text += `（${fields.join('、')}`
      if (targets.length > 3) text += `等${targets.length}个`
      text += '）'
    }
    text += '，基于关联特征可建立预测模型'
    result.push(text)
  }

  const numericVars = Object.keys(variableTypesData).filter(k => variableTypesData[k]?.type === 'continuous')
  const rows = dataShape.rows || 0
  if (numericVars.length >= 3 && rows >= 100) {
    result.push(`${numericVars.length}个数值指标，${rows}个样本，可识别分群`)
  }

  const categoricalVars = Object.keys(variableTypesData).filter(k =>
    ['categorical', 'categorical_numeric', 'ordinal'].includes(variableTypesData[k]?.type)
  )
  if (categoricalVars.length >= 3) {
    result.push(`${categoricalVars.length}个分类变量，可发现「如果A则B」的关联模式`)
  }

  return result
})

const hasDiscoveries = computed(() => {
  return qualityDiscoveries.value.length > 0 ||
         patternDiscoveries.value.length > 0 ||
         predictionDiscoveries.value.length > 0 ||
         numericCount.value > 0 ||
         categoricalCount.value > 0
})

// ==================== 加载数据 ====================
async function loadData() {
  let sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
  }
  if (!sessionId) {
    loading.value = false
    return
  }

  if (!sessionStore.currentSessionId) {
    sessionStore.currentSessionId = sessionId
  }

  loading.value = true
  try {
    const [reportResult, qualityResult] = await Promise.all([
      reportApi.get(sessionId),
      reportApi.getQuality(sessionId)
    ])

    reportData.value = reportResult
    qualityData.value = qualityResult

  } catch (err) {
    console.error('加载报告失败:', err)
    ElMessage.error('加载报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

// ✅ 新增：下载 HTML（前端动态生成）
async function handleDownloadHtml() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('没有可导出的会话')
    return
  }

  try {
    await reportApi.downloadHtml(sessionId)
    ElMessage.success('HTML 报告导出成功')
  } catch (err) {
    ElMessage.error('导出失败: ' + err.message)
  }
}

// ==================== 导出 ====================
async function handleExport(format) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('没有可导出的会话')
    return
  }

  try {
    const blob = await reportApi.export(sessionId, format)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report_${sessionId}.${format === 'html' ? 'html' : 'json'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    ElMessage.success(`导出成功: ${format.toUpperCase()}`)
  } catch (err) {
    ElMessage.error('导出失败: ' + err.message)
  }
}

function goTo(routeName) {
  router.push(`/${routeName}`)
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.report-summary {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px 20px 80px 20px;
}
.loading-container {
  padding: 40px 0;
}
.empty-state {
  padding: 60px 0;
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
  font-size: 20px;
  color: #2c3e50;
}
.header-actions {
  display: flex;
  gap: 8px;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 30px;
}
.stat-card {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 14px 16px;
  text-align: center;
  transition: all 0.2s;
}
.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.stat-value {
  font-size: 26px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
}

.grade-good .stat-value { color: #67c23a; }
.grade-warn .stat-value { color: #e6a23c; }
.grade-bad .stat-value { color: #f56c6c; }
.status-ok .stat-value { color: #67c23a; }
.status-warn .stat-value { color: #e6a23c; }
.status-bad .stat-value { color: #f56c6c; }
.status-high .stat-value { color: #409EFF; }

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
.chart-card.full-width {
  grid-column: 1 / -1;
  margin-bottom: 16px;
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
  height: 220px;
}
.chart-empty {
  height: 180px;
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
    height: 200px;
  }
}

.section {
  margin-top: 8px;
}
.section h3 {
  margin-bottom: 16px;
  color: #2c3e50;
  font-size: 18px;
}
.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
  background: #f5f7fa;
  border-radius: 8px;
}

.discovery-groups {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.discovery-group {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 16px 20px;
  border-left: 4px solid #909399;
}
.field-group { border-left-color: #409eff; }
.quality-group { border-left-color: #e6a23c; }
.pattern-group { border-left-color: #67c23a; }
.prediction-group { border-left-color: #9b59b6; }

.group-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.group-icon {
  font-size: 18px;
}
.group-title {
  font-weight: 600;
  font-size: 15px;
  color: #2c3e50;
}

.group-content {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 24px;
  padding-left: 4px;
}
.field-item {
  font-size: 14px;
  color: #555;
}
.field-label {
  color: #909399;
}
.field-value {
  font-weight: 600;
  color: #2c3e50;
}

.discovery-list {
  margin: 0;
  padding-left: 20px;
  list-style: none;
}
.discovery-list li {
  padding: 4px 0;
  font-size: 13px;
  color: #555;
  line-height: 1.6;
  position: relative;
  padding-left: 8px;
}
.discovery-list li::before {
  content: "•";
  color: #667eea;
  font-weight: bold;
  position: absolute;
  left: -12px;
}

@media (max-width: 768px) {
  .report-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .header-actions {
    width: 100%;
  }
  .header-actions .el-button {
    flex: 1;
  }
  .group-content {
    gap: 6px 12px;
  }
  .field-item {
    font-size: 13px;
  }
}
</style>