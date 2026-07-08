<template>
  <div class="data-overview">
    <h2>📊 数据概览</h2>
    <p class="subtitle">查看数据的基本信息和字段详情</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="overview-content">
      <!-- ===== 统计卡片 ===== -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.rows || 0 }}</div>
          <div class="stat-label">总行数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.columns || 0 }}</div>
          <div class="stat-label">总列数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ variableTypesCount }}</div>
          <div class="stat-label">变量类型</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ missingFieldsCount }}</div>
          <div class="stat-label">缺失字段</div>
        </div>
      </div>

      <!-- ===== 图表区域 ===== -->
      <div class="charts-row">
        <!-- 变量类型分布饼图 -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 变量类型分布</span>
          </div>
          <v-chart v-if="hasPieData" :option="pieOption" class="chart-container" />
          <div v-else class="chart-empty">暂无变量类型数据</div>
        </div>

        <!-- 分类变量类别数分布（过滤派生字段） -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 分类变量类别数</span>
          </div>
          <v-chart v-if="hasCategoryCountData" :key="'cat_count_' + catCountKey" :option="categoryCountOption" class="chart-container" />
          <div v-else class="chart-empty">暂无分类变量数据</div>
        </div>
      </div>

      <!-- 连续变量取值范围（min-max） -->
      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📊 连续变量取值范围</span>
        </div>
        <v-chart v-if="hasContinuousRangeData" :key="'cont_range_' + contRangeKey" :option="continuousRangeOption" class="chart-container" style="height: 280px;" />
        <div v-else class="chart-empty">暂无连续变量数据</div>
      </div>

      <!-- 缺失率最高的字段 -->
      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📋 缺失率最高的字段</span>
        </div>
        <v-chart v-if="hasMissingChartData" :option="missingBarOption" class="chart-container" style="height: 280px;" />
        <div v-else class="chart-empty">暂无缺失数据</div>
      </div>

      <!-- ===== 字段详情（三 Tab：连续变量 / 分类变量 / 其他类型） ===== -->
      <div class="section">
        <h4>📋 字段详情</h4>
        <el-tabs v-model="fieldTab" class="field-tabs">
          <!-- Tab1: 连续变量 -->
          <el-tab-pane label="连续变量" name="continuous">
            <el-table :data="continuousVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="120" fixed="left" show-overflow-tooltip />
              <el-table-column prop="count" label="样本量" width="80" align="center" />
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }">
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="mean" label="均值" width="100" align="center">
                <template #default="{ row }">
                  {{ row.mean !== undefined ? row.mean.toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="median" label="中位数" width="100" align="center">
                <template #default="{ row }">
                  {{ row.median !== undefined ? row.median.toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="std" label="标准差" width="100" align="center">
                <template #default="{ row }">
                  {{ row.std !== undefined ? row.std.toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="min" label="最小值" width="90" align="center">
                <template #default="{ row }">
                  {{ row.min !== undefined ? row.min.toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="max" label="最大值" width="90" align="center">
                <template #default="{ row }">
                  {{ row.max !== undefined ? row.max.toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <!-- 微图表：区间条 -->
              <el-table-column label="分布区间" min-width="160" align="center">
                <template #default="{ row }">
                  <div class="range-bar-wrapper" :title="`${row.min.toFixed(2)} ~ ${row.max.toFixed(2)}，均值 ${row.mean.toFixed(2)}`">
                    <div class="range-bar-track">
                      <div
                        class="range-bar-fill"
                        :style="{
                          left: getRangeBarLeft(row),
                          width: getRangeBarWidth(row)
                        }"
                      />
                      <div
                        class="range-bar-dot"
                        :style="{
                          left: getRangeBarDot(row)
                        }"
                      />
                    </div>
                    <div class="range-bar-labels">
                      <span class="range-min">{{ row.min !== undefined ? row.min.toFixed(1) : '-' }}</span>
                      <span class="range-max">{{ row.max !== undefined ? row.max.toFixed(1) : '-' }}</span>
                    </div>
                  </div>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="continuousVarList.length === 0" class="empty-tip">暂无连续变量</div>
          </el-tab-pane>

          <!-- Tab2: 分类变量 -->
          <el-tab-pane label="分类变量" name="categorical">
            <el-table :data="categoricalVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="120" fixed="left" show-overflow-tooltip />
              <el-table-column prop="count" label="样本量" width="80" align="center" />
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }">
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="n_unique" label="类别数" width="80" align="center" />
              <el-table-column prop="mode" label="众数" width="100" align="center" show-overflow-tooltip />
              <el-table-column prop="mode_pct" label="众数占比" width="90" align="center">
                <template #default="{ row }">
                  {{ row.mode_pct !== undefined ? row.mode_pct.toFixed(1) + '%' : '-' }}
                </template>
              </el-table-column>
              <!-- 微图表：进度条 Top3 -->
              <el-table-column label="类别分布 (Top3)" min-width="180">
                <template #default="{ row }">
                  <div v-if="row.topCategories && row.topCategories.length > 0" class="category-bars">
                    <div
                      v-for="(item, idx) in row.topCategories.slice(0, 3)"
                      :key="idx"
                      class="category-bar-item"
                    >
                      <span class="category-name">{{ item.name }}</span>
                      <div class="category-bar-track">
                        <div
                          class="category-bar-fill"
                          :style="{
                            width: item.pct + '%',
                            backgroundColor: getCategoryColor(idx)
                          }"
                        />
                      </div>
                      <span class="category-pct">{{ item.pct.toFixed(1) }}%</span>
                    </div>
                  </div>
                  <span v-else class="text-muted">-</span>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="categoricalVarList.length === 0" class="empty-tip">暂无分类变量</div>
          </el-tab-pane>

          <!-- Tab3: 其他类型 -->
          <el-tab-pane label="其他类型" name="other">
            <el-table :data="otherVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="140" fixed="left" show-overflow-tooltip />
              <el-table-column prop="type_desc" label="类型" width="100" align="center" />
              <el-table-column prop="count" label="样本量" width="80" align="center" />
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }">
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="key_info" label="关键信息" min-width="200" show-overflow-tooltip />
            </el-table>
            <div v-if="otherVarList.length === 0" class="empty-tip">暂无其他类型变量</div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>

    <div v-else-if="!loading" class="empty-state">
      <el-empty description="请先上传数据并完成分析">
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
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)
const fieldTab = ref('continuous')

const catCountKey = ref(0)
const contRangeKey = ref(0)

watch(() => reportData.value, () => {
  nextTick(() => {
    catCountKey.value += 1
    contRangeKey.value += 1
  })
}, { immediate: true })

const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

// ==================== 统计卡片 ====================
const typeCounts = computed(() => {
  const variableTypes = reportData.value?.variable_types || {}
  const counts = {}
  Object.values(variableTypes).forEach(info => {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  })
  return counts
})
const variableTypesCount = computed(() => Object.keys(typeCounts.value).length)
const missingFieldsCount = computed(() => reportData.value?.quality_report?.missing?.length || 0)

// ==================== 连续变量列表（带区间条数据） ====================
const continuousVarList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const result = []
  const allMin = []
  const allMax = []
  Object.entries(summaries).forEach(([name, info]) => {
    if (info.type === 'continuous') {
      const minVal = info.min !== undefined ? info.min : 0
      const maxVal = info.max !== undefined ? info.max : 0
      const meanVal = info.mean !== undefined ? info.mean : 0
      allMin.push(minVal)
      allMax.push(maxVal)
      result.push({
        name,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        mean: meanVal,
        median: info.median !== undefined ? info.median : 0,
        std: info.std !== undefined ? info.std : 0,
        min: minVal,
        max: maxVal,
        _min: minVal,
        _max: maxVal,
        _mean: meanVal
      })
    }
  })
  // 排序：按缺失率降序
  result.sort((a, b) => b.missing_pct - a.missing_pct)
  return result
})

// ==================== 分类变量列表（带进度条数据） ====================
const categoricalVarList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const result = []
  Object.entries(summaries).forEach(([name, info]) => {
    const typ = info.type
    if (typ === 'categorical' || typ === 'categorical_numeric' || typ === 'ordinal') {
      // 获取 top_categories 或 value_counts
      let topCats = []
      const total = info.count || 0
      if (info.top_categories && Object.keys(info.top_categories).length > 0) {
        const entries = Object.entries(info.top_categories)
        const totalCount = entries.reduce((s, e) => s + e[1], 0)
        topCats = entries.map(([cat, count]) => ({
          name: String(cat),
          count: count,
          pct: totalCount > 0 ? (count / totalCount * 100) : 0
        })).sort((a, b) => b.count - a.count)
      } else if (info.value_counts && Object.keys(info.value_counts).length > 0) {
        const entries = Object.entries(info.value_counts)
        const totalCount = entries.reduce((s, e) => s + e[1], 0)
        topCats = entries.map(([cat, count]) => ({
          name: String(cat),
          count: count,
          pct: totalCount > 0 ? (count / totalCount * 100) : 0
        })).sort((a, b) => b.count - a.count)
      }
      result.push({
        name,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        n_unique: info.n_unique || 0,
        mode: info.mode || '-',
        mode_pct: info.mode_pct || 0,
        topCategories: topCats.slice(0, 5)
      })
    }
  })
  result.sort((a, b) => b.missing_pct - a.missing_pct)
  return result
})

// ==================== 其他类型列表 ====================
const otherVarList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const result = []
  Object.entries(summaries).forEach(([name, info]) => {
    const typ = info.type
    if (typ === 'datetime' || typ === 'identifier' || typ === 'text') {
      let keyInfo = ''
      if (typ === 'datetime') {
        const minDate = info.min_date || info.min
        const maxDate = info.max_date || info.max
        if (minDate && maxDate) {
          keyInfo = `范围: ${minDate} ~ ${maxDate}`
        } else {
          keyInfo = `唯一值: ${info.n_unique || 0}`
        }
      } else if (typ === 'identifier') {
        keyInfo = `唯一值: ${info.n_unique || 0}`
      } else if (typ === 'text') {
        keyInfo = `唯一值: ${info.n_unique || 0}`
      }
      result.push({
        name,
        type_desc: info.type_desc || typ,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        key_info: keyInfo
      })
    }
  })
  result.sort((a, b) => b.missing_pct - a.missing_pct)
  return result
})

// ==================== 区间条辅助函数 ====================
function getRangeBarLeft(row) {
  const min = row._min !== undefined ? row._min : 0
  const max = row._max !== undefined ? row._max : 0
  const range = max - min
  if (range === 0) return '0%'
  return ((row._mean - min) / range * 100) + '%'
}

function getRangeBarWidth(row) {
  const min = row._min !== undefined ? row._min : 0
  const max = row._max !== undefined ? row._max : 0
  const range = max - min
  if (range === 0) return '100%'
  // 至少显示5%宽度，否则看不见
  const width = (range / (max + 1)) * 100
  return Math.max(width, 5) + '%'
}

function getRangeBarDot(row) {
  const min = row._min !== undefined ? row._min : 0
  const max = row._max !== undefined ? row._max : 0
  const range = max - min
  if (range === 0) return '50%'
  return ((row._mean - min) / range * 100) + '%'
}

function getCategoryColor(idx) {
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6']
  return colors[idx % colors.length]
}

// ==================== 饼图 ====================
const hasPieData = computed(() => Object.keys(typeCounts.value).length > 0)

const pieOption = computed(() => {
  const counts = typeCounts.value
  const colorMap = {
    continuous: '#409EFF', categorical: '#67C23A',
    categorical_numeric: '#E6A23C', ordinal: '#F56C6C',
    datetime: '#9B59B6', identifier: '#1ABC9C', text: '#95A5A6'
  }
  const data = Object.entries(counts).map(([type, count]) => ({
    name: typeDisplay[type] || type,
    value: count,
    itemStyle: { color: colorMap[type] || '#909399' }
  }))
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { orient: 'vertical', left: 'left', top: 'center', itemWidth: 12, itemHeight: 12 },
    series: [{
      type: 'pie',
      radius: ['40%', '65%'],
      center: ['55%', '50%'],
      avoidLabelOverlap: true,
      label: { show: true, formatter: '{d}%', fontSize: 11 },
      labelLine: { show: true },
      emphasis: { scale: true },
      data
    }]
  }
})

// ==================== 分类变量类别数分布 ====================
const derivedPatterns = ['_year', '_month', '_quarter', '_week', '_weekday', '_day', '_is_weekend']
const isDerivedField = (name) => derivedPatterns.some(p => name.endsWith(p))

const hasCategoryCountData = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    if (isDerivedField(key)) return false
    const info = summaries[key]
    return info.type && ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
  })
  return catVars.length > 0
})

const categoryCountOption = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    if (isDerivedField(key)) return false
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
    grid: { left: '12%', right: '8%', top: '15%', bottom: '20%' },
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

// ==================== 连续变量取值范围 ====================
const hasContinuousRangeData = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  return Object.keys(summaries).filter(key => summaries[key]?.type === 'continuous').length > 0
})

const continuousRangeOption = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const contVars = Object.keys(summaries).filter(key => summaries[key]?.type === 'continuous')
  if (contVars.length === 0) return {}

  const topVars = contVars.slice(0, 8)
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6', '#1ABC9C', '#3498DB', '#2ECC71']

  const data = topVars.map((key, idx) => {
    const info = summaries[key]
    const minVal = info.min !== undefined ? info.min : 0
    const maxVal = info.max !== undefined ? info.max : 0
    const meanVal = info.mean !== undefined ? info.mean : 0
    return {
      name: key,
      min: minVal,
      max: maxVal,
      mean: meanVal,
      color: colors[idx % colors.length]
    }
  })

  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        const d = data.find(item => item.name === p.name)
        if (d) {
          return `<strong>${d.name}</strong><br/>最小值：${d.min.toFixed(2)}<br/>最大值：${d.max.toFixed(2)}<br/>均值：${d.mean.toFixed(2)}`
        }
        return ''
      }
    },
    grid: { left: '12%', right: '8%', top: '10%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLabel: { rotate: 30, fontSize: 10, interval: 0 }
    },
    yAxis: { type: 'value', name: '取值范围' },
    series: [
      {
        type: 'bar',
        name: '取值范围',
        data: data.map(d => ({
          value: d.max - d.min,
          itemStyle: { color: d.color, opacity: 0.3 }
        })),
        barWidth: '40%',
        label: { show: true, position: 'top', formatter: function(params) {
          const d = data[params.dataIndex]
          return `${d.min.toFixed(1)} ~ ${d.max.toFixed(1)}`
        }, fontSize: 9 }
      }
    ]
  }
})

// ==================== 缺失率柱状图 ====================
const hasMissingChartData = computed(() => {
  const missing = reportData.value?.quality_report?.missing || []
  return missing.length > 0
})

const missingBarOption = computed(() => {
  const missing = reportData.value?.quality_report?.missing || []
  const sorted = [...missing].sort((a, b) => (b.percent || 0) - (a.percent || 0))
  const top = sorted.slice(0, 8)
  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>缺失率：${p.value}%`
      }
    },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: top.map(m => m.column || '未知'),
      axisLabel: { rotate: 30, fontSize: 10, interval: 0 },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      max: 100,
      name: '缺失率 (%)',
      nameLocation: 'middle',
      nameGap: 35,
      nameTextStyle: { fontSize: 11, color: '#909399' }
    },
    series: [{
      type: 'bar',
      data: top.map(m => ({
        value: Math.round(m.percent || 0),
        itemStyle: {
          color: (m.percent || 0) > 50 ? '#F56C6C' : (m.percent || 0) > 20 ? '#E6A23C' : '#67C23A'
        }
      })),
      barWidth: '50%',
      label: {
        show: true,
        position: 'top',
        formatter: '{c}%',
        fontSize: 10
      }
    }]
  }
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

  loading.value = true
  try {
    const result = await reportApi.get(sessionId)
    reportData.value = result
    console.log('✅ 数据概览加载完成')
  } catch (err) {
    console.error('加载数据失败:', err)
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
.data-overview {
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.stat-card {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}

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
@media (max-width: 768px) {
  .charts-row {
    grid-template-columns: 1fr;
  }
  .chart-container {
    height: 200px;
  }
}

.section {
  margin-top: 30px;
}
.section h4 {
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 16px;
}

.field-tabs {
  margin-top: 4px;
}

.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
}

/* ===== 区间条样式 ===== */
.range-bar-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 2px 0;
}

.range-bar-track {
  position: relative;
  width: 100%;
  height: 6px;
  background: #e8ecf1;
  border-radius: 3px;
}

.range-bar-fill {
  position: absolute;
  top: 0;
  height: 100%;
  background: rgba(64, 158, 255, 0.25);
  border-radius: 3px;
  min-width: 2px;
}

.range-bar-dot {
  position: absolute;
  top: -4px;
  width: 14px;
  height: 14px;
  background: #409EFF;
  border-radius: 50%;
  border: 2px solid white;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
  transform: translateX(-7px);
  cursor: pointer;
}

.range-bar-labels {
  display: flex;
  justify-content: space-between;
  width: 100%;
  font-size: 9px;
  color: #909399;
  margin-top: 2px;
}

.range-min, .range-max {
  font-size: 9px;
  color: #909399;
}

/* ===== 分类进度条样式 ===== */
.category-bars {
  display: flex;
  flex-direction: column;
  gap: 3px;
  width: 100%;
}

.category-bar-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}

.category-name {
  min-width: 20px;
  max-width: 40px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #555;
  font-size: 10px;
}

.category-bar-track {
  flex: 1;
  height: 4px;
  background: #e8ecf1;
  border-radius: 2px;
  overflow: hidden;
}

.category-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s;
  min-width: 2px;
}

.category-pct {
  font-size: 10px;
  color: #909399;
  min-width: 32px;
  text-align: right;
}

.text-muted {
  color: #bbb;
  font-size: 12px;
}
</style>