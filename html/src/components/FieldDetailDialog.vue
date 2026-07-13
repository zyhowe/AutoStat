// src/components/FieldDetailDialog.vue
<template>
  <el-dialog
    v-model="visible"
    :title="`🔍 字段详情：${fieldName}`"
    width="1000px"
    top="5vh"
    destroy-on-close
    :close-on-click-modal="true"
    @close="handleClose"
  >
    <div class="field-detail-body">
      <el-tabs v-model="activeTab" class="detail-tabs">
        <!-- ==================== Tab1: 概览 ==================== -->
        <el-tab-pane label="📊 概览" name="overview">
          <div class="detail-overview">
            <div class="overview-cards">
              <div class="ov-card"><span class="ov-label">类型</span><span class="ov-value">{{ fieldData.varTypeDesc || fieldData.varType || '-' }}</span></div>
              <div class="ov-card"><span class="ov-label">样本量</span><span class="ov-value">{{ fieldData.summary?.count || 0 }}</span></div>
              <div class="ov-card"><span class="ov-label">缺失率</span><span class="ov-value">{{ safePercent(fieldData.summary?.missing_pct) }}</span></div>
              <div class="ov-card">
                <span class="ov-label">{{ isContinuous ? '均值' : isCategorical ? '众数' : isDatetime ? '起始日期' : '中心值' }}</span>
                <span class="ov-value">
                  {{ isContinuous ? safeNumber(fieldData.summary?.mean) :
                     isCategorical ? (fieldData.summary?.mode || '-') :
                     isDatetime ? (fieldData.summary?.min_date || '-') : '-' }}
                </span>
              </div>
              <div class="ov-card">
                <span class="ov-label">{{ isContinuous ? '范围' : isCategorical ? '类别数' : isDatetime ? '时间跨度' : '范围' }}</span>
                <span class="ov-value">
                  {{ isContinuous ? safeRange(fieldData.summary?.min, fieldData.summary?.max) :
                     isCategorical ? (fieldData.summary?.n_unique || 0) :
                     isDatetime ? (fieldData.summary?.date_range_days || 0) + '天' : '-' }}
                </span>
              </div>
            </div>
            <div class="ov-chart">
              <div v-if="hasDistributionChart" class="chart-wrapper">
                <v-chart :option="distributionChartOption" class="dist-chart" autoresize />
                <div class="chart-tip">📌 蓝色条形为实际数据分布，红色虚线为均值位置</div>
              </div>
              <div v-else class="ov-chart-empty">暂无分布图表</div>
            </div>
          </div>
        </el-tab-pane>

        <!-- ==================== Tab2: 统计详情 ==================== -->
        <el-tab-pane label="📈 统计详情" name="stats">
          <div class="detail-stats">
            <div v-if="isContinuous">
              <el-descriptions :column="3" border size="small">
                <el-descriptions-item label="样本量">{{ fieldData.summary?.count || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失数">{{ fieldData.summary?.missing || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失率">{{ safePercent(fieldData.summary?.missing_pct) }}</el-descriptions-item>
                <el-descriptions-item label="均值">{{ safeNumber(fieldData.summary?.mean) }}</el-descriptions-item>
                <el-descriptions-item label="中位数">{{ safeNumber(fieldData.summary?.median) }}</el-descriptions-item>
                <el-descriptions-item label="标准差">{{ safeNumber(fieldData.summary?.std) }}</el-descriptions-item>
                <el-descriptions-item label="最小值">{{ safeNumber(fieldData.summary?.min) }}</el-descriptions-item>
                <el-descriptions-item label="最大值">{{ safeNumber(fieldData.summary?.max) }}</el-descriptions-item>
                <el-descriptions-item label="极差">{{ safeRange(fieldData.summary?.min, fieldData.summary?.max) }}</el-descriptions-item>
                <el-descriptions-item label="Q1 (25%)">{{ safeNumber(fieldData.summary?.q1) }}</el-descriptions-item>
                <el-descriptions-item label="Q3 (75%)">{{ safeNumber(fieldData.summary?.q3) }}</el-descriptions-item>
                <el-descriptions-item label="IQR">{{ safeRange(fieldData.summary?.q1, fieldData.summary?.q3) }}</el-descriptions-item>
                <el-descriptions-item label="偏度">{{ safeNumber(fieldData.summary?.skew) }}</el-descriptions-item>
                <el-descriptions-item label="峰度">{{ safeNumber(fieldData.summary?.kurtosis) }}</el-descriptions-item>
                <el-descriptions-item label="正态性">{{ fieldData.summary?.is_normal ? '✅ 是' : '❌ 否' }}</el-descriptions-item>
              </el-descriptions>
              <div v-if="hasBoxPlotData" class="stats-chart">
                <v-chart :option="boxPlotOption" class="stats-chart-container" autoresize />
              </div>
            </div>
            <div v-else-if="isCategorical">
              <el-descriptions :column="3" border size="small">
                <el-descriptions-item label="样本量">{{ fieldData.summary?.count || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失数">{{ fieldData.summary?.missing || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失率">{{ safePercent(fieldData.summary?.missing_pct) }}</el-descriptions-item>
                <el-descriptions-item label="类别数">{{ fieldData.summary?.n_unique || 0 }}</el-descriptions-item>
                <el-descriptions-item label="众数">{{ fieldData.summary?.mode || '-' }}</el-descriptions-item>
                <el-descriptions-item label="众数频数">{{ fieldData.summary?.mode_freq || 0 }}</el-descriptions-item>
                <el-descriptions-item label="众数占比" :span="1">{{ safePercent(fieldData.summary?.mode_pct) }}</el-descriptions-item>
              </el-descriptions>
              <div v-if="fieldData.topCategories && fieldData.topCategories.length > 0" class="category-full-list">
                <h5>类别分布</h5>
                <div class="category-bars-full">
                  <div v-for="(item, idx) in fieldData.topCategories" :key="idx" class="category-bar-item-full">
                    <span class="category-name-full">{{ item.name }}</span>
                    <div class="category-bar-track-full">
                      <div class="category-bar-fill-full" :style="{ width: item.pct + '%', backgroundColor: getCategoryColor(idx) }" />
                    </div>
                    <span class="category-pct-full">{{ safePercent(item.pct) }} ({{ item.count }})</span>
                  </div>
                </div>
              </div>
            </div>
            <div v-else-if="isDatetime">
              <el-descriptions :column="3" border size="small">
                <el-descriptions-item label="样本量">{{ fieldData.summary?.count || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失数">{{ fieldData.summary?.missing || 0 }}</el-descriptions-item>
                <el-descriptions-item label="缺失率">{{ safePercent(fieldData.summary?.missing_pct) }}</el-descriptions-item>
                <el-descriptions-item label="起始日期">{{ fieldData.summary?.min_date || '-' }}</el-descriptions-item>
                <el-descriptions-item label="结束日期">{{ fieldData.summary?.max_date || '-' }}</el-descriptions-item>
                <el-descriptions-item label="时间跨度">{{ fieldData.summary?.date_range_days || 0 }} 天</el-descriptions-item>
                <el-descriptions-item label="唯一日期数">{{ fieldData.summary?.n_unique || 0 }}</el-descriptions-item>
              </el-descriptions>
            </div>
            <div v-else>
              <el-empty description="该类型暂无详细统计信息" :image-size="60" />
            </div>
          </div>
        </el-tab-pane>

        <!-- ==================== Tab3: 时序 ==================== -->
        <el-tab-pane label="📅 时序" name="timeseries">
          <div v-if="hasTimeseriesData" class="detail-timeseries">
            <div class="ts-diagnosis-cards">
              <div class="ts-card"><span class="ts-label">样本数</span><span class="ts-value">{{ fieldData.tsDiag?.n_samples || 0 }}</span></div>
              <div class="ts-card"><span class="ts-label">平稳性</span><span class="ts-value" :class="fieldData.tsDiag?.is_stationary ? 'pass' : 'fail'">{{ fieldData.tsDiag?.is_stationary ? '✅ 平稳' : '⚠️ 非平稳' }}</span></div>
              <div class="ts-card"><span class="ts-label">自相关性</span><span class="ts-value" :class="fieldData.tsDiag?.has_autocorrelation ? 'pass' : 'fail'">{{ fieldData.tsDiag?.has_autocorrelation ? '✅ 有' : '❌ 无' }}</span></div>
              <div class="ts-card"><span class="ts-label">季节性</span><span class="ts-value" :class="fieldData.tsDiag?.has_seasonality ? 'pass' : 'fail'">{{ fieldData.tsDiag?.has_seasonality ? '✅ 有' : '❌ 无' }}</span></div>
              <div class="ts-card"><span class="ts-label">均值</span><span class="ts-value">{{ safeNumber(fieldData.tsDiag?.mean) }}</span></div>
              <div class="ts-card"><span class="ts-label">标准差</span><span class="ts-value">{{ safeNumber(fieldData.tsDiag?.std) }}</span></div>
            </div>
            <div v-if="hasTimeseriesChartData" class="timeseries-chart">
              <v-chart :option="timeseriesChartOption" class="timeseries-chart-container" autoresize />
            </div>
            <div v-if="hasACFData" class="acf-chart">
              <v-chart :option="acfChartOption" class="acf-chart-container" autoresize />
            </div>
          </div>
          <div v-else class="detail-empty">⏳ 该字段无时间序列数据</div>
        </el-tab-pane>

        <!-- ==================== Tab4: 质量 ==================== -->
        <el-tab-pane label="⭐ 质量" name="quality">
          <div class="detail-quality">
            <div class="quality-grid">
              <div class="quality-card completeness">
                <div class="qc-title">📋 完整性</div>
                <div v-if="fieldData.missing" class="qc-content">
                  <span>缺失数：{{ fieldData.missing.count || 0 }}</span>
                  <span>缺失率：{{ safePercent(fieldData.missing.percent) }}</span>
                </div>
                <div v-else class="qc-empty">✅ 无缺失值</div>
              </div>
              <div class="quality-card accuracy">
                <div class="qc-title">🎯 准确性</div>
                <div v-if="fieldData.outlier" class="qc-content">
                  <span>异常数：{{ fieldData.outlier.count || 0 }}</span>
                  <span>异常率：{{ safePercent(fieldData.outlier.percent) }}</span>
                  <span>下界：{{ safeNumber(fieldData.outlier.lower_bound) }}</span>
                  <span>上界：{{ safeNumber(fieldData.outlier.upper_bound) }}</span>
                </div>
                <div v-else class="qc-empty">✅ 无异常值</div>
              </div>
              <div class="quality-card consistency">
                <div class="qc-title">🔗 一致性</div>
                <div v-if="fieldData.rules && fieldData.rules.length > 0" class="qc-content">
                  <span>参与勾稽规则：{{ fieldData.rules.length }} 条</span>
                  <el-button size="small" text type="primary" @click="activeTab = 'rules'">查看详情 →</el-button>
                </div>
                <div v-else class="qc-empty">✅ 无勾稽规则</div>
              </div>
              <div class="quality-card uniqueness">
                <div class="qc-title">🆔 唯一性</div>
                <div v-if="fieldData.duplicateInfo" class="qc-content">
                  <span>重复数：{{ fieldData.duplicateInfo.count || 0 }}</span>
                  <span>重复率：{{ safePercent(fieldData.duplicateInfo.percent) }}</span>
                </div>
                <div v-else class="qc-empty">✅ 无重复记录</div>
              </div>
            </div>
          </div>
        </el-tab-pane>

        <!-- ==================== Tab5: 相关性 ==================== -->
        <el-tab-pane label="🔗 相关性" name="correlation">
          <div class="detail-correlation">
            <div v-if="fieldData.correlations && fieldData.correlations.length > 0" class="corr-list">
              <div v-for="(item, idx) in fieldData.correlations" :key="idx" class="corr-item">
                <span class="corr-var">{{ item.var }}</span>
                <span class="corr-value" :style="{ color: Math.abs(item.value) >= 0.9 ? '#F56C6C' : '#E6A23C' }">
                  r = {{ safeNumber(item.value) }}
                </span>
                <span class="corr-strength">{{ getStrengthLabel(item.value) }}</span>
              </div>
              <div v-if="fieldData.correlations.length > 1" class="corr-chart">
                <v-chart :option="corrBarChartOption" class="corr-chart-container" autoresize />
              </div>
            </div>
            <div v-else class="detail-empty">暂无强相关关系 (|r| ≥ 0.7)</div>
          </div>
        </el-tab-pane>

        <!-- ==================== Tab6: 规则 ==================== -->
        <el-tab-pane label="📋 规则" name="rules">
          <div v-if="fieldData.rules && fieldData.rules.length > 0" class="detail-rules">
            <div class="rules-list">
              <div v-for="(rule, idx) in fieldData.rules" :key="idx" class="rule-item">
                <span class="rule-type">{{ rule.type }}</span>
                <code class="rule-expr">{{ rule.rule }}</code>
                <span class="rule-conf">{{ safePercent(rule.confidence) }}</span>
              </div>
            </div>
          </div>
          <div v-else class="detail-empty">📋 该字段未参与任何勾稽规则</div>
        </el-tab-pane>

        <!-- ==================== Tab7: 模型 ==================== -->
        <el-tab-pane label="🤖 模型" name="models">
          <div v-if="fieldData.models && fieldData.models.length > 0" class="detail-models">
            <div class="models-list">
              <div v-for="(model, idx) in fieldData.models" :key="idx" class="model-item">
                <span class="model-role">{{ model.role }}</span>
                <span class="model-task">{{ model.task_type }}</span>
                <span class="model-name">{{ model.model }}</span>
                <span class="model-target">{{ model.target }}</span>
              </div>
            </div>
          </div>
          <div v-else class="detail-empty">🤖 该字段未参与任何模型推荐</div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-dialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import VChart from 'vue-echarts'

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  fieldName: {
    type: String,
    default: ''
  },
  fieldData: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['update:modelValue', 'close'])

const visible = ref(false)
const activeTab = ref('overview')

watch(() => props.modelValue, (val) => {
  visible.value = val
  if (val) {
    activeTab.value = 'overview'
  }
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

function handleClose() {
  emit('close')
  emit('update:modelValue', false)
}

// ==================== 类型安全辅助函数 ====================
function safeNumber(value, decimals = 2) {
  if (value === undefined || value === null) return '-'
  const num = Number(value)
  if (isNaN(num)) return '-'
  if (!Number.isFinite(num)) return '-'
  return num.toFixed(decimals)
}

function safePercent(value) {
  if (value === undefined || value === null) return '0%'
  const num = Number(value)
  if (isNaN(num)) return '0%'
  return num.toFixed(1) + '%'
}

function safeRange(min, max) {
  const minNum = Number(min)
  const maxNum = Number(max)
  if (isNaN(minNum) || isNaN(maxNum)) return '-'
  return minNum.toFixed(2) + ' ~ ' + maxNum.toFixed(2)
}

// ===== 类型判断 =====
const isContinuous = computed(() => props.fieldData?.varType === 'continuous')
const isCategorical = computed(() => props.fieldData?.varType === 'categorical' || props.fieldData?.varType === 'categorical_numeric' || props.fieldData?.varType === 'ordinal')
const isDatetime = computed(() => props.fieldData?.varType === 'datetime')

// ============================================================
// Tab1: 概览 - 分布图
// ============================================================
const hasDistributionChart = computed(() => {
  if (isContinuous.value) {
    const min = props.fieldData?.summary?.min
    const max = props.fieldData?.summary?.max
    return min !== undefined && min !== null && max !== undefined && max !== null
  }
  if (isCategorical.value) {
    return props.fieldData?.topCategories && props.fieldData.topCategories.length > 0
  }
  return false
})

const distributionChartOption = computed(() => {
  const summary = props.fieldData?.summary || {}

  if (isContinuous.value && summary.min !== undefined && summary.min !== null && summary.max !== undefined && summary.max !== null) {
    const min = Number(summary.min) || 0
    const max = Number(summary.max) || 0
    const mean = Number(summary.mean) || 0
    const median = Number(summary.median) || 0
    const std = Number(summary.std) || 1
    const bins = 20
    const binWidth = (max - min) / bins

    const data = []
    for (let i = 0; i < bins; i++) {
      const x = min + i * binWidth + binWidth / 2
      const z = (x - mean) / std
      const pdf = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI))
      data.push([x, Math.round(pdf * summary.count * binWidth * 0.8 + 1)])
    }

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          const p = params[0]
          return `<strong>${p.name}</strong><br/>频数：${Math.round(p.value)}`
        }
      },
      grid: { left: '10%', right: '8%', top: '12%', bottom: '18%' },
      xAxis: {
        type: 'category',
        data: data.map(d => d[0].toFixed(1)),
        axisLabel: { rotate: 20, fontSize: 9, interval: 2 },
        name: '值',
        nameLocation: 'center',
        nameGap: 30,
        nameTextStyle: { fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        name: '频数',
        nameLocation: 'center',
        nameGap: 40,
        nameTextStyle: { fontSize: 11 }
      },
      series: [{
        type: 'bar',
        data: data.map(d => ({ value: Math.round(d[1]), itemStyle: { color: '#409EFF', opacity: 0.7 } })),
        barWidth: '55%'
      }],
      markLine: {
        silent: true,
        symbol: 'none',
        data: [
          { name: '均值', xAxis: mean.toFixed(1), label: { formatter: `均值: ${mean.toFixed(1)}`, color: '#F56C6C', fontSize: 10 } },
          { name: '中位数', xAxis: median.toFixed(1), label: { formatter: `中位数: ${median.toFixed(1)}`, color: '#67C23A', fontSize: 10 } }
        ],
        lineStyle: { type: 'dashed' }
      }
    }
  }

  if (isCategorical.value && props.fieldData?.topCategories) {
    const data = props.fieldData.topCategories.slice(0, 10)
    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          const p = params[0]
          return `<strong>${p.name}</strong><br/>数量：${p.value}`
        }
      },
      grid: { left: '14%', right: '8%', top: '12%', bottom: '20%' },
      xAxis: {
        type: 'category',
        data: data.map(d => d.name),
        axisLabel: { rotate: 20, fontSize: 10, interval: 1 },
        name: '类别',
        nameLocation: 'center',
        nameGap: 35,
        nameTextStyle: { fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        name: '数量',
        nameLocation: 'center',
        nameGap: 40,
        nameTextStyle: { fontSize: 11 }
      },
      series: [{
        type: 'bar',
        data: data.map((d, i) => ({ value: d.count, itemStyle: { color: getCategoryColor(i) } })),
        barWidth: '45%',
        label: { show: true, position: 'top', formatter: '{c}', fontSize: 10 }
      }]
    }
  }

  return {}
})

// ============================================================
// Tab2: 统计详情 - 箱线图
// ============================================================
const hasBoxPlotData = computed(() => {
  return isContinuous.value && props.fieldData?.summary?.q1 !== undefined && props.fieldData?.summary?.q1 !== null
})

const boxPlotOption = computed(() => {
  const summary = props.fieldData?.summary || {}
  const min = Number(summary.min) || 0
  const q1 = Number(summary.q1) || 0
  const median = Number(summary.median) || 0
  const q3 = Number(summary.q3) || 0
  const max = Number(summary.max) || 0
  return {
    tooltip: { trigger: 'item' },
    grid: { left: '12%', right: '10%', top: '10%', bottom: '18%' },
    xAxis: { type: 'category', data: [props.fieldName], axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', name: '值', nameTextStyle: { fontSize: 11 }, nameLocation: 'center', nameGap: 40 },
    series: [{
      type: 'boxplot',
      data: [[min, q1, median, q3, max]],
      itemStyle: { color: '#409EFF' }
    }]
  }
})

// ============================================================
// Tab3: 时序
// ============================================================
const hasTimeseriesData = computed(() => {
  return props.fieldData?.tsDiag !== null && props.fieldData?.tsDiag !== undefined
})

const hasTimeseriesChartData = computed(() => {
  const points = props.fieldData?.tsDiag?.data_points
  return points && points.length > 2
})

const timeseriesChartOption = computed(() => {
  const points = props.fieldData?.tsDiag?.data_points || []
  if (points.length < 2) return {}
  const dates = points.map(p => p.date || '')
  const values = points.map(p => Number(p.value) || 0)
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>值：${p.value?.toFixed(2)}`
      }
    },
    grid: { left: '10%', right: '8%', top: '12%', bottom: '18%' },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: { rotate: 20, fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 20)) },
      name: '时间',
      nameLocation: 'center',
      nameGap: 35,
      nameTextStyle: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: '值',
      nameLocation: 'center',
      nameGap: 40,
      nameTextStyle: { fontSize: 11 }
    },
    series: [{
      type: 'line',
      data: values,
      smooth: true,
      lineStyle: { color: '#409EFF', width: 2 },
      areaStyle: { color: 'rgba(64,158,255,0.2)' },
      symbol: 'circle',
      symbolSize: 6,
      itemStyle: { color: '#409EFF' }
    }]
  }
})

// ===== ACF 图 =====
const hasACFData = computed(() => {
  const points = props.fieldData?.tsDiag?.data_points
  return points && points.length > 5
})

const acfChartOption = computed(() => {
  const points = props.fieldData?.tsDiag?.data_points || []
  if (points.length < 5) return {}

  const values = points.map(p => Number(p.value) || 0)
  const n = values.length
  const maxLag = Math.min(20, n - 1)
  const mean = values.reduce((s, v) => s + v, 0) / n

  const acfValues = []
  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < n - lag; i++) {
      sum += (values[i] - mean) * (values[i + lag] - mean)
      sumSq += (values[i] - mean) ** 2
    }
    if (lag === 0) {
      acfValues.push(1)
    } else {
      acfValues.push(sum / sumSq)
    }
  }

  const lags = acfValues.map((_, i) => i)
  const ci = 1.96 / Math.sqrt(n)

  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0]
        return `<strong>滞后 ${p.name}</strong><br/>自相关系数：${p.value.toFixed(3)}`
      }
    },
    grid: { left: '12%', right: '8%', top: '12%', bottom: '18%' },
    xAxis: {
      type: 'category',
      data: lags,
      axisLabel: { fontSize: 10 },
      name: '滞后阶数 (Lag)',
      nameLocation: 'center',
      nameGap: 35,
      nameTextStyle: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      min: -1,
      max: 1,
      name: '自相关系数',
      nameLocation: 'center',
      nameGap: 40,
      nameTextStyle: { fontSize: 11 },
      splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } }
    },
    series: [{
      type: 'bar',
      data: acfValues.map((v, i) => ({
        value: v,
        itemStyle: {
          color: Math.abs(v) > ci ? (v > 0 ? '#F56C6C' : '#67C23A') : '#C0C4CC'
        }
      })),
      barWidth: '55%',
      markLine: {
        silent: true,
        symbol: 'none',
        data: [
          { yAxis: ci, label: { formatter: `95% CI (${ci.toFixed(3)})`, color: '#409EFF', fontSize: 10 } },
          { yAxis: -ci, label: { formatter: `-95% CI (-${ci.toFixed(3)})`, color: '#409EFF', fontSize: 10 } }
        ],
        lineStyle: { color: '#409EFF', type: 'dashed' }
      }
    }]
  }
})

// ============================================================
// 辅助方法
// ============================================================
function getCategoryColor(idx) {
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6', '#1ABC9C', '#3498DB', '#2ECC71', '#E67E22', '#E74C3C']
  return colors[idx % colors.length]
}

function getStrengthLabel(value) {
  const abs = Math.abs(Number(value) || 0)
  if (abs >= 0.9) return '极强'
  if (abs >= 0.7) return '强'
  if (abs >= 0.5) return '中'
  return '弱'
}

// corrBarChartOption 简单实现
const corrBarChartOption = computed(() => {
  const correlations = props.fieldData?.correlations || []
  if (correlations.length < 2) return {}
  const data = correlations.slice(0, 10).map(item => ({
    name: item.var,
    value: Number(item.value) || 0
  }))
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>相关系数：${p.value.toFixed(3)}`
      }
    },
    grid: { left: '14%', right: '8%', top: '10%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLabel: { rotate: 20, fontSize: 10, interval: 1 }
    },
    yAxis: {
      type: 'value',
      name: '相关系数',
      min: -1,
      max: 1,
      nameLocation: 'center',
      nameGap: 40,
      nameTextStyle: { fontSize: 11 }
    },
    series: [{
      type: 'bar',
      data: data.map(d => ({
        value: d.value,
        itemStyle: { color: Math.abs(d.value) >= 0.7 ? '#F56C6C' : '#E6A23C' }
      })),
      barWidth: '45%',
      label: { show: true, position: 'top', formatter: (p) => p.value.toFixed(3), fontSize: 9 }
    }]
  }
})

defineExpose({
  open: () => {
    visible.value = true
  }
})
</script>

<style scoped>
.field-detail-body {
  height: 68vh;
  overflow-y: auto;
  padding-right: 6px;
}
.field-detail-body::-webkit-scrollbar {
  width: 6px;
}
.field-detail-body::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}
.field-detail-body::-webkit-scrollbar-track {
  background: #f0f2f6;
  border-radius: 3px;
}

.detail-tabs :deep(.el-tabs__item) {
  font-size: 13px;
}
.detail-tabs :deep(.el-tabs__header) {
  margin-bottom: 12px;
}

.overview-cards {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}
.ov-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.ov-label {
  font-size: 11px;
  color: #909399;
}
.ov-value {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 2px;
}
.ov-chart {
  background: #fafafa;
  border-radius: 8px;
  padding: 12px;
  min-height: 220px;
}
.dist-chart {
  width: 100%;
  height: 260px;
}
.ov-chart-empty {
  text-align: center;
  padding: 40px 0;
  color: #bbb;
  font-size: 13px;
}
.chart-tip {
  font-size: 11px;
  color: #909399;
  text-align: center;
  margin-top: 6px;
}
.chart-wrapper {
  width: 100%;
}

.detail-stats {
  padding: 4px 0;
}
.stats-chart {
  margin-top: 16px;
  background: #fafafa;
  border-radius: 8px;
  padding: 12px;
}
.stats-chart-container {
  width: 100%;
  height: 200px;
}

.category-full-list {
  margin-top: 16px;
}
.category-full-list h5 {
  margin-bottom: 8px;
  color: #2c3e50;
  font-size: 14px;
}
.category-bars-full {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.category-bar-item-full {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.category-name-full {
  min-width: 60px;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #555;
}
.category-bar-track-full {
  flex: 1;
  height: 8px;
  background: #e8ecf1;
  border-radius: 4px;
  overflow: hidden;
}
.category-bar-fill-full {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
  min-width: 2px;
}
.category-pct-full {
  min-width: 80px;
  text-align: right;
  color: #909399;
  font-size: 12px;
}

.detail-timeseries {
  padding: 4px 0;
}
.ts-diagnosis-cards {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}
.ts-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.ts-label {
  font-size: 11px;
  color: #909399;
}
.ts-value {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 2px;
}
.ts-value.pass {
  color: #67c23a;
}
.ts-value.fail {
  color: #f56c6c;
}
.timeseries-chart {
  background: #fafafa;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 16px;
}
.timeseries-chart-container {
  width: 100%;
  height: 220px;
}
.acf-chart {
  background: #fafafa;
  border-radius: 8px;
  padding: 12px;
}
.acf-chart-container {
  width: 100%;
  height: 200px;
}

.detail-quality {
  padding: 4px 0;
}
.quality-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.quality-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
}
.quality-card .qc-title {
  font-weight: 600;
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 8px;
}
.quality-card .qc-content {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 13px;
  color: #555;
}
.quality-card .qc-content span {
  background: white;
  padding: 4px 12px;
  border-radius: 4px;
}
.quality-card .qc-empty {
  color: #67c23a;
  font-size: 13px;
}
.quality-card.completeness {
  border-left: 4px solid #E6A23C;
}
.quality-card.accuracy {
  border-left: 4px solid #F56C6C;
}
.quality-card.consistency {
  border-left: 4px solid #409EFF;
}
.quality-card.uniqueness {
  border-left: 4px solid #67C23A;
}

.detail-correlation {
  padding: 4px 0;
}
.corr-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 16px;
}
.corr-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 6px 12px;
  background: #f5f7fa;
  border-radius: 6px;
}
.corr-var {
  font-weight: 500;
  color: #2c3e50;
  min-width: 120px;
}
.corr-value {
  font-weight: 600;
  min-width: 80px;
}
.corr-strength {
  font-size: 12px;
  color: #909399;
}
.corr-chart {
  background: #fafafa;
  border-radius: 8px;
  padding: 12px;
}
.corr-chart-container {
  width: 100%;
  height: 280px;
}

.detail-rules {
  padding: 4px 0;
}
.rules-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.rule-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  font-size: 13px;
}
.rule-type {
  font-size: 11px;
  color: #909399;
  min-width: 60px;
}
.rule-expr {
  flex: 1;
  font-family: 'Consolas', monospace;
  font-size: 12px;
  color: #2c3e50;
  background: white;
  padding: 2px 8px;
  border-radius: 4px;
}
.rule-conf {
  font-weight: 500;
  color: #67c23a;
  min-width: 50px;
  text-align: right;
}

.detail-models {
  padding: 4px 0;
}
.models-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.model-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  font-size: 13px;
}
.model-role {
  font-size: 14px;
  min-width: 50px;
}
.model-task {
  color: #409EFF;
  min-width: 80px;
}
.model-name {
  flex: 1;
  font-weight: 500;
}
.model-target {
  color: #909399;
  font-size: 12px;
}

.detail-empty {
  padding: 40px 0;
  text-align: center;
  color: #bbb;
  font-size: 14px;
}

@media (max-width: 768px) {
  .overview-cards {
    grid-template-columns: repeat(3, 1fr);
  }
  .ts-diagnosis-cards {
    grid-template-columns: repeat(3, 1fr);
  }
  .quality-grid {
    grid-template-columns: 1fr;
  }
  .field-detail-body {
    height: 75vh;
  }
  .dist-chart {
    height: 200px;
  }
  .timeseries-chart-container {
    height: 180px;
  }
  .acf-chart-container {
    height: 160px;
  }
  .corr-chart-container {
    height: 220px;
  }
  .stats-chart-container {
    height: 160px;
  }
}
</style>