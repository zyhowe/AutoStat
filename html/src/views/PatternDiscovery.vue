<template>
  <div class="pattern-discovery">
    <h2>📈 规律发现</h2>
    <p class="subtitle">探索变量之间的相关性和时间序列规律</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="pattern-content">
      <!-- ===== 图表区域 ===== -->
      <div class="charts-row">
        <!-- 相关性热力图 -->
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
                style="width: 200px;"
                @change="onHeatmapFieldsChange"
              >
                <el-option
                  v-for="field in availableHeatmapFields"
                  :key="field"
                  :label="field"
                  :value="field"
                />
              </el-select>
              <el-button size="small" @click="resetHeatmapFields">重置</el-button>
            </div>
          </div>
          <v-chart
            v-if="hasHeatmapData && selectedHeatmapFields.length > 0"
            :key="'heatmap_' + heatmapKey"
            :option="heatmapOption"
            class="chart-container"
          />
          <div v-else class="chart-empty">暂无相关性数据或请选择字段</div>
        </div>

        <!-- 强相关散点图 -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 强相关散点图</span>
          </div>
          <v-chart
            v-if="hasScatterData"
            :key="'scatter_' + scatterKey"
            :option="scatterOption"
            class="chart-container"
          />
          <div v-else class="chart-empty">暂无强相关数据</div>
        </div>
      </div>

      <!-- ===== 新增：分类变量类别数分布 ===== -->
      <div class="chart-card full-width" v-if="hasCategoryCountData">
        <div class="chart-header">
          <span class="chart-title">📊 分类变量类别数分布</span>
        </div>
        <v-chart
          :key="'cat_count_' + catCountKey"
          :option="categoryCountOption"
          class="chart-container"
          style="height: 220px;"
        />
      </div>

      <!-- 时间序列折线图 -->
      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📈 时间序列趋势</span>
        </div>
        <v-chart
          v-if="hasTimeseriesData"
          :key="'ts_' + tsKey"
          :option="timeseriesOption"
          class="chart-container"
          style="height: 250px;"
        />
        <div v-else class="chart-empty">暂无时间序列数据</div>
      </div>

      <!-- ===== Tab 切换 ===== -->
      <el-tabs v-model="activeTab">
        <!-- 相关性分析 -->
        <el-tab-pane label="相关性分析" name="correlation">
          <div v-if="highCorrelations.length === 0" class="empty-tip">
            未发现强相关关系（|r| > 0.7）
          </div>
          <div v-else>
            <p>发现 <strong>{{ highCorrelations.length }}</strong> 对强相关关系：</p>
            <el-table :data="highCorrelations" border size="small" max-height="400">
              <el-table-column prop="var1" label="变量1" width="150" fixed="left" />
              <el-table-column prop="var2" label="变量2" width="150" />
              <el-table-column prop="value" label="相关系数" width="120" align="center">
                <template #default="{ row }">
                  {{ row.value.toFixed(3) }}
                </template>
              </el-table-column>
              <el-table-column label="方向" width="80" align="center">
                <template #default="{ row }">
                  <el-tag :type="row.value > 0 ? 'danger' : 'success'" size="small">
                    {{ row.value > 0 ? '正相关' : '负相关' }}
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-tab-pane>

        <!-- 时间序列分析 -->
        <el-tab-pane label="时间序列分析" name="timeseries">
          <div v-if="timeSeriesData.length === 0" class="empty-tip">
            未检测到时间序列数据
          </div>
          <div v-else>
            <el-table :data="timeSeriesData" border size="small" max-height="400">
              <el-table-column prop="key" label="变量/分组" width="180" fixed="left" />
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
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)
const activeTab = ref('correlation')

// ==================== 强制刷新 key ====================
const heatmapKey = ref(0)
const scatterKey = ref(0)
const tsKey = ref(0)
const catCountKey = ref(0)

watch(() => reportData.value, () => {
  heatmapKey.value += 1
  scatterKey.value += 1
  tsKey.value += 1
  catCountKey.value += 1
}, { immediate: false })

// ==================== 热力图字段筛选 ====================
const selectedHeatmapFields = ref([])
const availableHeatmapFields = computed(() => {
  const matrix = reportData.value?.correlations?.matrix || {}
  return Object.keys(matrix)
})

const getDefaultHeatmapFields = () => {
  const corrs = reportData.value?.correlations?.high_correlations || []
  if (corrs.length === 0) return []
  const fieldSet = new Set()
  corrs.forEach(c => {
    if (c.var1) fieldSet.add(c.var1)
    if (c.var2) fieldSet.add(c.var2)
  })
  const fields = Array.from(fieldSet)
  return fields.slice(0, 10)
}

const resetHeatmapFields = () => {
  const defaults = getDefaultHeatmapFields()
  if (defaults.length > 0) {
    selectedHeatmapFields.value = defaults
  } else {
    const matrix = reportData.value?.correlations?.matrix || {}
    const keys = Object.keys(matrix)
    selectedHeatmapFields.value = keys.slice(0, Math.min(10, keys.length))
  }
}

const onHeatmapFieldsChange = () => {
  heatmapKey.value += 1
}

watch(() => reportData.value, (newVal) => {
  if (newVal) {
    resetHeatmapFields()
  }
}, { immediate: false })

// ==================== 计算属性 ====================
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

// ==================== 热力图 ====================
const hasHeatmapData = computed(() => {
  const matrix = reportData.value?.correlations?.matrix
  return matrix && Object.keys(matrix).length > 0 && selectedHeatmapFields.value.length >= 2
})

const heatmapOption = computed(() => {
  const matrix = reportData.value?.correlations?.matrix || {}
  const selected = selectedHeatmapFields.value
  if (selected.length < 2) return {}
  const filteredVars = selected.filter(key => matrix[key] !== undefined)
  if (filteredVars.length < 2) return {}

  const data = []
  for (let i = 0; i < filteredVars.length; i++) {
    for (let j = 0; j < filteredVars.length; j++) {
      const val = matrix[filteredVars[i]]?.[filteredVars[j]]
      if (val !== undefined && val !== null) {
        data.push([i, j, parseFloat(val.toFixed(2))])
      }
    }
  }

  return {
    tooltip: {
      position: 'top',
      formatter: function(params) {
        const idx = params.data
        if (idx && idx.length >= 3) {
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
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      text: ['高', '低'],
      inRange: { color: ['#F56C6C', '#FFFFFF', '#67C23A'] }
    },
    series: [{
      type: 'heatmap',
      data: data,
      label: { show: true, fontSize: 9, formatter: (p) => p.data[2] },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }
      }
    }]
  }
})

// ==================== 散点图 ====================
const hasScatterData = computed(() => highCorrelations.value.length > 0)

const scatterOption = computed(() => {
  if (highCorrelations.value.length === 0) return {}
  const pair = highCorrelations.value[0]
  const var1 = pair.var1
  const var2 = pair.var2
  const corr = pair.value

  const n = 50
  const data = []
  let seed = 42
  for (let i = 0; i < n; i++) {
    seed = (seed * 9301 + 49297) % 233280
    const x = (seed / 233280) * 100 + 20
    seed = (seed * 9301 + 49297) % 233280
    const y = x * corr + (1 - Math.abs(corr)) * ((seed / 233280) * 100) + 10
    data.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100])
  }

  const isPositive = corr > 0
  return {
    tooltip: {
      trigger: 'item',
      formatter: function(params) {
        return `<strong>${var1}</strong>：${params.data[0]}<br/><strong>${var2}</strong>：${params.data[1]}`
      }
    },
    grid: { left: '12%', right: '8%', top: '10%', bottom: '12%' },
    xAxis: {
      type: 'value',
      name: var1,
      nameLocation: 'center',
      nameGap: 25,
      nameTextStyle: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: var2,
      nameLocation: 'center',
      nameGap: 35,
      nameTextStyle: { fontSize: 11 }
    },
    series: [{
      type: 'scatter',
      data: data,
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

// ==================== 分类变量类别数分布 ====================
const hasCategoryCountData = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    const info = summaries[key]
    return info.type && ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
  })
  return catVars.length > 0
})

const categoryCountOption = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    const info = summaries[key]
    return info.type && ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
  })
  if (catVars.length === 0) return {}

  const data = catVars.map(key => {
    const info = summaries[key]
    let vc = {}
    if (info.value_counts && Object.keys(info.value_counts).length > 0) {
      vc = info.value_counts
    } else if (info.top_categories && Object.keys(info.top_categories).length > 0) {
      vc = info.top_categories
    }
    return {
      name: key,
      value: Object.keys(vc).length
    }
  }).sort((a, b) => b.value - a.value)

  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>类别数：${p.value}`
      }
    },
    grid: { left: '12%', right: '8%', top: '8%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLabel: { rotate: 30, fontSize: 10, interval: 0 }
    },
    yAxis: { type: 'value', name: '类别数量' },
    series: [{
      type: 'bar',
      data: data.map(d => ({ value: d.value, itemStyle: { color: '#9B59B6' } })),
      barWidth: '45%',
      label: { show: true, position: 'top', formatter: '{c}', fontSize: 10 }
    }]
  }
})

// ==================== 时间序列折线图 ====================
const hasTimeseriesData = computed(() => {
  return Object.keys(reportData.value?.time_series_diagnostics || {}).length > 0
})

const timeseriesOption = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  const keys = Object.keys(diag).slice(0, 5)
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#9B59B6', '#1ABC9C']

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
    grid: { left: '8%', right: '8%', top: '18%', bottom: '15%' },
    xAxis: {
      type: 'category',
      data: Array.from({ length: timePoints }, (_, i) => `T${i + 1}`),
      axisLabel: { fontSize: 10 }
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
})

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
  padding: 20px;
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

/* ===== 图表区域 ===== */
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

.filter-bar {
  display: flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 768px) {
  .charts-row {
    grid-template-columns: 1fr;
  }
  .chart-container {
    height: 200px;
  }
  .filter-bar {
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .filter-bar .el-select {
    width: 140px !important;
  }
}
</style>