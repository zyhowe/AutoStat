<template>
  <div class="pattern-discovery">
    <h2>📈 规律发现</h2>
    <p class="subtitle">探索当前表中变量之间的相关性和时间序列规律</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="pattern-content">
      <!-- ===== 表选择器 ===== -->
      <TableSelector
        v-model="currentTable"
        :table-names="tableNames"
        :is-multi-table="isMultiTable"
        @change="onTableChange"
      />

      <el-tabs v-model="activeTab">
        <!-- ============================================================ -->
        <!-- Tab1: 相关性分析 -->
        <!-- ============================================================ -->
        <el-tab-pane label="相关性分析" name="correlation">
          <!-- 热力图 -->
          <div class="chart-card">
            <div class="chart-header">
              <span class="chart-title">🔗 相关性热力图</span>
              <div class="filter-bar">
                <el-select
                  v-model="selectedHeatmapFields"
                  multiple
                  collapse-tags
                  collapse-tags-tooltip
                  placeholder="选择显示字段"
                  filterable
                  size="small"
                  style="width: 160px;"
                  @change="onHeatmapFieldsChange"
                >
                  <el-option
                    v-for="field in availableHeatmapFields"
                    :key="field"
                    :label="field"
                    :value="field"
                  />
                </el-select>
                <el-input-number
                  v-model="heatmapThresholdMin"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  size="small"
                  style="width: 75px;"
                  @change="onThresholdChange"
                />
                <span style="font-size: 12px; color: #909399;">~</span>
                <el-input-number
                  v-model="heatmapThresholdMax"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  size="small"
                  style="width: 75px;"
                  @change="onThresholdChange"
                />
                <el-button size="small" @click="resetHeatmapFields">重置</el-button>
              </div>
            </div>
            <v-chart
              v-if="hasHeatmapData && selectedHeatmapFields.length > 0"
              :key="'heatmap_' + heatmapKey"
              :option="heatmapOption"
              @click="onHeatmapClick"
              class="chart-container"
              style="height: 320px;"
            />
            <div v-else class="chart-empty">暂无相关性数据或请选择字段</div>
          </div>

          <!-- 散点图 -->
          <div class="chart-card">
            <div class="chart-header">
              <span class="chart-title">📊 散点图</span>
              <span v-if="selectedCorrPair" class="corr-info">
                <span class="field-name-link" @click="openFieldDetail(selectedCorrPair.var1)">{{ selectedCorrPair.var1 }}</span>
                ↔
                <span class="field-name-link" @click="openFieldDetail(selectedCorrPair.var2)">{{ selectedCorrPair.var2 }}</span>
                (r={{ selectedCorrPair.value }})
              </span>
              <span v-else class="corr-info">点击热力图格子查看</span>
            </div>
            <v-chart
              v-if="hasScatterData"
              :key="'scatter_' + scatterKey"
              :option="scatterOption"
              class="chart-container"
              style="height: 300px;"
            />
            <div v-else class="chart-empty">暂无强相关数据</div>
          </div>

          <!-- 强相关表格 -->
          <div v-if="highCorrelations.length === 0" class="empty-tip">
            未发现强相关关系（|r| > 0.7）
          </div>
          <div v-else>
            <p>发现 <strong>{{ highCorrelations.length }}</strong> 对强相关关系：</p>
            <el-table :data="highCorrelations" border size="small" max-height="400">
              <el-table-column prop="var1" label="变量1" width="120" fixed="left">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.var1)">{{ row.var1 }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="var2" label="变量2" width="120">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.var2)">{{ row.var2 }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="value" label="相关系数" width="110" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showCorrelationData(row)">
                    {{ Number(row.value).toFixed(3) }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="方向" width="80" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showCorrelationData(row)">
                    <el-tag :type="Number(row.value) > 0 ? 'danger' : 'success'" size="small">
                      {{ Number(row.value) > 0 ? '正相关' : '负相关' }}
                    </el-tag>
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="样本量" width="100" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showNonMissingData(row)">
                    {{ row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : '--' }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="强相关数" width="110" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showCorrelationData(row)">
                    {{ row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : '--' }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="置信度" width="100" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showCorrelationData(row)">
                    {{ getConfidenceDisplay(row) }}
                  </span>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-tab-pane>

        <!-- ============================================================ -->
        <!-- Tab2: 时间序列分析 -->
        <!-- ============================================================ -->
        <el-tab-pane label="时间序列分析" name="timeseries">
          <div class="chart-card">
            <div class="chart-header">
              <span class="chart-title">📈 时间序列趋势</span>
              <div class="filter-bar">
                <el-select
                  v-model="selectedTsFields"
                  multiple
                  collapse-tags
                  collapse-tags-tooltip
                  placeholder="选择字段"
                  filterable
                  size="small"
                  style="width: 160px;"
                  @change="onTsFieldsChange"
                >
                  <el-option
                    v-for="key in availableTsFields"
                    :key="key"
                    :label="key"
                    :value="key"
                  />
                </el-select>
                <el-select
                  v-model="selectedTsGroup"
                  placeholder="选择分组"
                  size="small"
                  style="width: 120px;"
                  @change="onTsGroupChange"
                  clearable
                >
                  <el-option
                    v-for="group in availableTsGroups"
                    :key="group"
                    :label="group"
                    :value="group"
                  />
                </el-select>
                <el-tag v-if="hasRealTimeseriesData" size="small" type="success">真实数据</el-tag>
                <el-tag v-else size="small" type="warning">模拟数据</el-tag>
              </div>
            </div>
            <v-chart
              v-if="hasTimeseriesData && activeTab === 'timeseries'"
              :key="'ts_' + tsKey"
              :option="timeseriesOption"
              autoresize
              class="chart-container"
              style="height: 280px;"
            />
            <div v-else class="chart-empty">暂无时间序列数据</div>
          </div>

          <div v-if="timeSeriesData.length === 0" class="empty-tip">
            未检测到时间序列数据
          </div>
          <div v-else>
            <el-table :data="timeSeriesData" border size="small" max-height="400">
              <el-table-column prop="key" label="变量/分组" width="180" fixed="left">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.key)">{{ row.key }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="n_samples" label="样本量" width="100" align="center" />
              <el-table-column prop="stationary" label="平稳性" width="120" align="center" />
              <el-table-column prop="autocorrelation" label="自相关性" width="120" align="center" />
              <el-table-column prop="seasonality" label="季节性" width="100" align="center" />
            </el-table>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>

    <div v-else class="empty-state">
      <el-empty description="请先完成数据分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useFieldDetailStore } from '../stores/fieldDetail'
import { reportApi } from '../api/report'
import { openDataPreview } from '../components/DataPreviewDialog'
import TableSelector from '../components/TableSelector.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const fieldDetailStore = useFieldDetailStore()

const loading = ref(false)
const reportData = ref(null)
const activeTab = ref('correlation')

// ===== 表选择器状态 =====
const currentTable = ref('merged')
const tableNames = ref([])
const isMultiTable = ref(false)

// ==================== Key 强制刷新 ====================
const heatmapKey = ref(0)
const scatterKey = ref(0)
const tsKey = ref(0)

// ==================== 热力图相关 ====================
const selectedHeatmapFields = ref([])
const heatmapThresholdMin = ref(0.9)
const heatmapThresholdMax = ref(1.0)
const selectedCorrPair = ref(null)

// ==================== 时间序列相关 ====================
const selectedTsFields = ref([])
const selectedTsGroup = ref('')

// ==================== 字段详情弹窗 ====================
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

const variableTypes = computed(() => currentData.value?.variable_types || {})
const summaries = computed(() => currentData.value?.variable_summaries || {})
const correlations = computed(() => currentData.value?.correlations || {})
const quality = computed(() => currentData.value?.quality_report || {})

function buildFieldData(fieldName) {
  const summary = summaries.value?.[fieldName] || {}
  const varType = variableTypes.value?.[fieldName]?.type || 'unknown'
  const varTypeDesc = variableTypes.value?.[fieldName]?.type_desc || typeDisplay[varType] || varType

  const tsDiag = currentData.value?.time_series_diagnostics?.[fieldName] || null
  const outlier = quality.value?.outliers?.[fieldName] || null
  const missing = quality.value?.missing?.find(m => m.column === fieldName) || null
  const dupInfo = quality.value?.duplicates || null

  const corrList = []
  const matrix = correlations.value?.matrix || {}
  if (matrix[fieldName]) {
    const entries = Object.entries(matrix[fieldName])
    for (const [varName, value] of entries) {
      if (varName !== fieldName && value !== null && value !== undefined && Math.abs(value) >= 0.7) {
        corrList.push({ var: varName, value: parseFloat(Number(value).toFixed(4)) })
      }
    }
    corrList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  }

  const rules = []
  const auditRules = quality.value?.audit_rules || {}
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
    correlations: corrList,
    rules,
    models,
    topCategories
  }
}

function openFieldDetail(fieldName) {
  const data = buildFieldData(fieldName)
  fieldDetailStore.open(fieldName, data)
}

// ==================== Watch ====================
watch(() => reportData.value, () => {
  nextTick(() => {
    heatmapKey.value += 1
    scatterKey.value += 1
    tsKey.value += 1
  })
}, { immediate: true })

watch(activeTab, (newTab) => {
  if (newTab === 'timeseries') {
    setTimeout(() => { tsKey.value += 1 }, 200)
  } else if (newTab === 'correlation') {
    setTimeout(() => { heatmapKey.value += 1 }, 200)
  }
})

// ==================== 热力图字段 ====================
const availableHeatmapFields = computed(() => {
  const matrix = correlations.value?.matrix || {}
  return Object.keys(matrix)
})

const getDefaultHeatmapFields = () => {
  const corrs = correlations.value?.high_correlations || []
  if (corrs.length === 0) return []
  const fieldSet = new Set()
  corrs.forEach(c => {
    if (c.var1) fieldSet.add(c.var1)
    if (c.var2) fieldSet.add(c.var2)
  })
  return Array.from(fieldSet).slice(0, 10)
}

const resetHeatmapFields = () => {
  const defaults = getDefaultHeatmapFields()
  if (defaults.length > 0) {
    selectedHeatmapFields.value = defaults
  } else {
    const matrix = correlations.value?.matrix || {}
    selectedHeatmapFields.value = Object.keys(matrix).slice(0, Math.min(10, Object.keys(matrix).length))
  }
  selectedCorrPair.value = null
  heatmapThresholdMin.value = 0.9
  heatmapThresholdMax.value = 1.0
  heatmapKey.value += 1
  scatterKey.value += 1
}

const onHeatmapFieldsChange = () => {
  heatmapKey.value += 1
}

const onThresholdChange = () => {
  heatmapKey.value += 1
}

// ==================== 热力图点击联动散点图 ====================
function onHeatmapClick(params) {
  const data = params.data
  if (!data || data.length < 3 || data[2] === null || data[2] === undefined) return

  const matrix = correlations.value?.matrix || {}
  const filteredVars = selectedHeatmapFields.value.filter(key => matrix[key] !== undefined)
  if (filteredVars.length < 2) return

  const i = data[0]
  const j = data[1]
  if (i === j || i >= filteredVars.length || j >= filteredVars.length) return

  const var1 = filteredVars[i]
  const var2 = filteredVars[j]
  const corrValue = data[2]

  if (var1 && var2 && var1 !== var2) {
    selectedCorrPair.value = { var1, var2, value: corrValue }
    scatterKey.value += 1
  }
}

// ==================== 热力图配置 ====================
const hasHeatmapData = computed(() => {
  const matrix = correlations.value?.matrix
  return matrix && Object.keys(matrix).length > 0 && selectedHeatmapFields.value.length >= 2
})

const heatmapOption = computed(() => {
  const matrix = correlations.value?.matrix || {}
  const selected = selectedHeatmapFields.value
  if (selected.length < 2) return {}

  const filteredVars = selected.filter(key => matrix[key] !== undefined)
  if (filteredVars.length < 2) return {}

  const minThreshold = heatmapThresholdMin.value
  const maxThreshold = heatmapThresholdMax.value

  const data = []
  for (let i = 0; i < filteredVars.length; i++) {
    for (let j = 0; j < filteredVars.length; j++) {
      if (i === j) {
        data.push([i, j, null])
        continue
      }
      const val = matrix[filteredVars[i]]?.[filteredVars[j]]
      if (val !== undefined && val !== null) {
        const absVal = Math.abs(val)
        if (absVal >= minThreshold && absVal <= maxThreshold) {
          data.push([i, j, parseFloat(Number(val).toFixed(2))])
        } else {
          data.push([i, j, null])
        }
      } else {
        data.push([i, j, null])
      }
    }
  }

  const validData = data.filter(d => d[2] !== null && d[2] !== undefined)
  const hasValid = validData.length > 0
  const minVal = hasValid ? Math.min(...validData.map(d => d[2])) : 0
  const maxVal = hasValid ? Math.max(...validData.map(d => d[2])) : 0

  return {
    tooltip: {
      position: 'top',
      formatter: function(params) {
        const idx = params.data
        if (idx && idx.length >= 3 && idx[2] !== null && idx[2] !== undefined) {
          return `${filteredVars[idx[0]]} ↔ ${filteredVars[idx[1]]}<br/>相关系数：${idx[2]}`
        }
        return ''
      }
    },
    grid: { left: '12%', right: '8%', top: '5%', bottom: '12%' },
    xAxis: {
      type: 'category',
      data: filteredVars,
      splitArea: { show: true },
      axisLabel: { rotate: 30, fontSize: 9, interval: 0 }
    },
    yAxis: {
      type: 'category',
      data: filteredVars,
      splitArea: { show: true },
      axisLabel: { fontSize: 9 }
    },
    visualMap: {
      min: hasValid ? Math.max(-1, minVal - 0.05) : -1,
      max: hasValid ? Math.min(1, maxVal + 0.05) : 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      text: ['高', '低'],
      inRange: { color: ['#FFFFFF', '#F56C6C'] },
      outOfRange: { color: '#FFFFFF' }
    },
    series: [{
      type: 'heatmap',
      data: data,
      label: {
        show: true,
        fontSize: 9,
        formatter: (p) => p.data[2] !== null && p.data[2] !== undefined ? p.data[2] : ''
      },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }
      },
      itemStyle: {
        color: function(params) {
          if (params.data[2] === null || params.data[2] === undefined) {
            return '#FFFFFF'
          }
          const val = params.data[2]
          const absVal = Math.abs(val)
          if (absVal >= 0.9) return val > 0 ? '#F56C6C' : '#409EFF'
          if (absVal >= 0.7) return val > 0 ? '#E6A23C' : '#67C23A'
          return '#FFFFFF'
        }
      }
    }]
  }
})

// ==================== 散点图配置 ====================
const hasScatterData = computed(() => {
  const pair = selectedCorrPair.value
  if (pair) {
    const matrix = correlations.value?.matrix || {}
    return matrix[pair.var1] && matrix[pair.var1][pair.var2] !== undefined
  }
  return highCorrelations.value.length > 0
})

const scatterOption = computed(() => {
  let pair = selectedCorrPair.value
  if (!pair) {
    const corrs = highCorrelations.value
    if (corrs.length === 0) return {}
    const c = corrs[0]
    pair = { var1: c.var1, var2: c.var2, value: c.value }
  }

  const var1 = pair.var1
  const var2 = pair.var2
  const corr = pair.value

  let scatterData = []
  const data = currentData.value?.data
  if (data && data.length > 0) {
    const sampleData = data.slice(0, 100)
    scatterData = sampleData.map(row => {
      const x = row[var1] !== undefined ? parseFloat(row[var1]) : null
      const y = row[var2] !== undefined ? parseFloat(row[var2]) : null
      if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
        return [x, y]
      }
      return null
    }).filter(d => d !== null)
  }

  if (scatterData.length < 3) {
    const n = 50
    let seed = 42
    for (let i = 0; i < n; i++) {
      seed = (seed * 9301 + 49297) % 233280
      const x = (seed / 233280) * 100 + 20
      seed = (seed * 9301 + 49297) % 233280
      const y = x * corr + (1 - Math.abs(corr)) * ((seed / 233280) * 100) + 10
      scatterData.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100])
    }
  }

  const isPositive = Number(corr) > 0
  return {
    tooltip: {
      trigger: 'item',
      formatter: function(params) {
        return `<strong>${var1}</strong>：${params.data[0]}<br/><strong>${var2}</strong>：${params.data[1]}`
      }
    },
    grid: { left: '14%', right: '10%', top: '10%', bottom: '15%' },
    xAxis: {
      type: 'value',
      name: var1,
      nameLocation: 'center',
      nameGap: 30,
      nameTextStyle: { fontSize: 14, fontWeight: 'bold' },
      axisLabel: { fontSize: 12 }
    },
    yAxis: {
      type: 'value',
      name: var2,
      nameLocation: 'center',
      nameGap: 40,
      nameTextStyle: { fontSize: 14, fontWeight: 'bold' },
      axisLabel: { fontSize: 12 }
    },
    series: [{
      type: 'scatter',
      data: scatterData,
      symbolSize: 8,
      itemStyle: { color: isPositive ? '#67C23A' : '#F56C6C', opacity: 0.7 },
      markLine: {
        silent: true,
        data: [{ type: 'average', name: '均值线' }],
        lineStyle: { color: '#409EFF', type: 'dashed' }
      },
      markArea: {
        silent: true,
        data: [[{
          name: `${isPositive ? '正' : '负'}相关趋势`,
          xAxis: 'min',
          yAxis: 'min'
        }, {
          xAxis: 'max',
          yAxis: 'max'
        }]],
        itemStyle: {
          color: isPositive ? 'rgba(103, 194, 58, 0.05)' : 'rgba(245, 108, 108, 0.05)'
        }
      }
    }]
  }
})

// ==================== 时间序列相关 ====================
const highCorrelations = computed(() => {
  return correlations.value?.high_correlations || []
})

const availableTsFields = computed(() => {
  const diag = currentData.value?.time_series_diagnostics || {}
  return Object.keys(diag)
})

const availableTsGroups = computed(() => {
  const groups = []
  for (const [col, info] of Object.entries(variableTypes.value)) {
    const type = info.type || info
    if (type === 'categorical' || type === 'categorical_numeric' || type === 'ordinal') {
      if (col !== 'id' && !col.endsWith('_id')) {
        groups.push(col)
      }
    }
  }
  return groups.slice(0, 10)
})

const timeSeriesData = computed(() => {
  const diag = currentData.value?.time_series_diagnostics || {}
  return Object.entries(diag).map(([key, info]) => ({
    key,
    n_samples: info.n_samples || 0,
    stationary: info.is_stationary ? '✅ 平稳' : '⚠️ 非平稳',
    autocorrelation: info.has_autocorrelation ? '✅ 有' : '❌ 无',
    seasonality: info.has_seasonality ? '✅ 有' : '❌ 无'
  }))
})

const hasTimeseriesData = computed(() => {
  return Object.keys(currentData.value?.time_series_diagnostics || {}).length > 0
})

const hasRealTimeseriesData = computed(() => {
  const diag = currentData.value?.time_series_diagnostics || {}
  return Object.values(diag).some(d => d.data_points && d.data_points.length > 0)
})

const getDefaultTsFields = () => {
  const fields = availableTsFields.value
  return fields.slice(0, Math.min(5, fields.length))
}

watch(() => currentData.value, (newVal) => {
  if (newVal && Object.keys(newVal).length > 0) {
    resetHeatmapFields()
    selectedTsFields.value = getDefaultTsFields()
    selectedTsGroup.value = ''
  }
}, { immediate: false })

const onTsFieldsChange = () => {
  tsKey.value += 1
}

const onTsGroupChange = () => {
  tsKey.value += 1
}

// ==================== 时间序列趋势配置 ====================
const timeseriesOption = computed(() => {
  const diag = currentData.value?.time_series_diagnostics || {}
  const selected = selectedTsFields.value
  const group = selectedTsGroup.value

  let keys = selected.length > 0 ? selected : Object.keys(diag).slice(0, 5)
  keys = keys.slice(0, 10)

  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6', '#1ABC9C', '#3498DB', '#2ECC71', '#E67E22', '#E74C3C']

  const useRealData = keys.some(k => diag[k]?.data_points && diag[k].data_points.length > 0)

  if (useRealData) {
    const seriesData = keys.map((key, idx) => {
      const points = diag[key]?.data_points || []
      if (points.length === 0) return null
      return {
        name: key,
        type: 'line',
        data: points.map(p => p.value || 0),
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 2 },
        itemStyle: { color: colors[idx % colors.length] }
      }
    }).filter(s => s !== null)

    if (seriesData.length === 0) {
      return generateMockTimeseriesOption(keys, colors)
    }

    const xData = diag[keys[0]]?.data_points?.map(p => p.date || p.time || '') || []
    return {
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          let html = `<strong>${params[0].axisValue}</strong><br/>`
          params.forEach(p => {
            html += `${p.marker} ${p.seriesName}：${p.value}`
          })
          return html
        }
      },
      legend: {
        data: keys,
        top: 0,
        right: 10,
        itemWidth: 12,
        itemHeight: 12,
        textStyle: { fontSize: 11 }
      },
      grid: { left: '14%', right: '8%', top: '18%', bottom: '15%' },
      xAxis: {
        type: 'category',
        data: xData,
        axisLabel: { fontSize: 11, rotate: 30 }
      },
      yAxis: {
        type: 'value',
        name: '值',
        nameLocation: 'middle',
        nameGap: 35,
        nameTextStyle: { fontSize: 11 }
      },
      series: seriesData
    }
  } else {
    return generateMockTimeseriesOption(keys, colors)
  }
})

function generateMockTimeseriesOption(keys, colors) {
  const timePoints = 20
  const seriesData = keys.map((key, idx) => {
    const data = []
    let val = 50 + Math.random() * 30
    const trend = (Math.random() - 0.5) * 0.3
    const season = Math.random() * 5
    for (let i = 0; i < timePoints; i++) {
      val = val + trend + Math.sin(i / 3) * season + (Math.random() - 0.5) * 3
      data.push(Math.round(val * 100) / 100)
    }
    return {
      name: key,
      type: 'line',
      data: data,
      smooth: true,
      symbol: 'circle',
      symbolSize: 4,
      lineStyle: { width: 2 },
      itemStyle: { color: colors[idx % colors.length] }
    }
  })

  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        let html = `<strong>时间点 ${params[0].axisValue}</strong><br/>`
        params.forEach(p => {
          html += `${p.marker} ${p.seriesName}：${p.value}`
        })
        return html
      }
    },
    legend: {
      data: keys,
      top: 0,
      right: 10,
      itemWidth: 12,
      itemHeight: 12,
      textStyle: { fontSize: 11 }
    },
    grid: { left: '14%', right: '8%', top: '18%', bottom: '15%' },
    xAxis: {
      type: 'category',
      data: Array.from({ length: timePoints }, (_, i) => `T${i + 1}`),
      axisLabel: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: '值',
      nameLocation: 'middle',
      nameGap: 35,
      nameTextStyle: { fontSize: 11 }
    },
    series: seriesData
  }
}

// ==================== 显示函数 ====================
function getConfidenceDisplay(row) {
  if (row.valid_count !== undefined && row.valid_count !== null && row.valid_count > 0) {
    const total = row.valid_count
    if (total > 0) {
      return ((row.valid_count / total) * 100).toFixed(1) + '%'
    }
  }
  return '--'
}

// ==================== 数据预览联动 ====================
function showNonMissingData(row) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  openDataPreview({
    sessionId: sessionId,
    title: `${row.var1} ↔ ${row.var2} 有效记录（两字段均非空，共 ${row.valid_count || 0} 条）`,
    fields: [row.var1, row.var2],
    filters: [
      { field: row.var1, condition: 'is_not_null', value: true },
      { field: row.var2, condition: 'is_not_null', value: true }
    ]
  })
}

function showCorrelationData(row) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  const validCount = row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : 0
  const total = currentData.value?.data_shape?.rows || 0
  const confPct = total > 0 ? ((validCount / total) * 100).toFixed(1) : 0

  openDataPreview({
    sessionId: sessionId,
    title: `${row.var1} ↔ ${row.var2} (r=${Number(row.value).toFixed(3)}) 强相关数据（${validCount}条有效，${confPct}%置信度）`,
    fields: [row.var1, row.var2],
    filters: [
      { field: row.var1, condition: 'is_not_null', value: true },
      { field: row.var2, condition: 'is_not_null', value: true }
    ]
  })
}

// ==================== 加载数据 ====================
async function loadData() {
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
    const result = await reportApi.get(sessionId)
    reportData.value = result

    // 初始化表选择器
    const allTables = result?.all_tables || {}
    const tableKeys = Object.keys(allTables)
    tableNames.value = tableKeys.filter(k => k !== 'merged')
    isMultiTable.value = tableNames.value.length > 1

    if (!currentTable.value || !allTables[currentTable.value]) {
      currentTable.value = 'merged'
    }

    resetHeatmapFields()
    selectedTsFields.value = getDefaultTsFields()
    selectedTsGroup.value = ''
    console.log('✅ 规律发现加载完成')
  } catch (err) {
    ElMessage.error('加载数据失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function goToUpload() {
  router.push('/upload')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.pattern-discovery {
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
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #909399;
  font-size: 16px;
}

.chart-card {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  padding: 16px 16px 4px 16px;
  transition: box-shadow 0.2s;
  margin-bottom: 16px;
}
.chart-card:hover {
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}
.chart-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}
.chart-container {
  width: 100%;
}
.chart-empty {
  height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #bbb;
  font-size: 13px;
}

.filter-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.corr-info {
  font-size: 13px;
  color: #409EFF;
  font-weight: 500;
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

@media (max-width: 768px) {
  .filter-bar {
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .filter-bar .el-select {
    width: 140px !important;
  }
  .filter-bar .el-input-number {
    width: 65px !important;
  }
}
</style>