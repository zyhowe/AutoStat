<template>
  <div class="data-overview">
    <h2>📊 数据概览</h2>
    <p class="subtitle">查看数据的基本信息和字段详情</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="overview-content">
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

        <!-- 分类变量分布柱状图 -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📋 分类变量分布</span>
          </div>
          <v-chart v-if="hasCategoricalChartData" :option="categoricalBarOption" class="chart-container" />
          <div v-else class="chart-empty">
            {{ catChartEmptyReason }}
          </div>
        </div>
      </div>

      <!-- 连续变量分布直方图 -->
      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📊 连续变量分布（前3个）</span>
        </div>
        <v-chart v-if="hasContinuousChartData" :option="continuousHistOption" class="chart-container" style="height: 280px;" />
        <div v-else class="chart-empty">暂无连续变量数据</div>
      </div>

      <!-- 变量类型分布 -->
      <div class="section">
        <h4>📋 变量类型分布</h4>
        <div class="type-tags">
          <el-tag
            v-for="(count, type) in typeCounts"
            :key="type"
            size="large"
            class="type-tag"
          >
            {{ typeDisplay[type] || type }}：{{ count }}
          </el-tag>
        </div>
      </div>

      <!-- 变量详情表格 -->
      <div class="section">
        <h4>📋 字段详情</h4>
        <el-table :data="variableList" border size="small" max-height="420" style="width: 100%;">
          <el-table-column prop="name" label="字段名" width="140" fixed="left" show-overflow-tooltip />
          <el-table-column prop="type_desc" label="类型" width="100" align="center" />
          <el-table-column prop="count" label="样本量" width="90" align="center" />
          <el-table-column prop="missing" label="缺失数" width="90" align="center" />
          <el-table-column prop="missing_pct" label="缺失率" width="90" align="center">
            <template #default="{ row }">
              {{ row.missing_pct.toFixed(1) }}%
            </template>
          </el-table-column>
          <el-table-column prop="center" label="中心趋势" width="120" align="center" show-overflow-tooltip />
          <el-table-column prop="spread" label="分布" min-width="120" show-overflow-tooltip />
        </el-table>
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
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)

const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

// ==================== 统计 ====================
const typeCounts = computed(() => {
  const variableTypes = reportData.value?.variable_types || {}
  const counts = {}
  for (const info of Object.values(variableTypes)) {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  }
  return counts
})

const variableTypesCount = computed(() => Object.keys(typeCounts.value).length)

const missingFieldsCount = computed(() => {
  return reportData.value?.quality_report?.missing?.length || 0
})

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
  })).slice(0, 30)
})

// ==================== 饼图数据 ====================
const hasPieData = computed(() => {
  return Object.keys(typeCounts.value).length > 0
})

const pieOption = computed(() => {
  const counts = typeCounts.value
  const colorMap = {
    continuous: '#409EFF',
    categorical: '#67C23A',
    categorical_numeric: '#E6A23C',
    ordinal: '#F56C6C',
    datetime: '#9B59B6',
    identifier: '#1ABC9C',
    text: '#95A5A6'
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
      data: data
    }]
  }
})

// ==================== 分类变量柱状图（优先显示类别数最多的） ====================
const catChartEmptyReason = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    const info = summaries[key]
    if (!info.type || !['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)) return false
    const hasValueCounts = info.value_counts && Object.keys(info.value_counts).length > 0
    const hasTopCategories = info.top_categories && Object.keys(info.top_categories).length > 0
    return hasValueCounts || hasTopCategories
  })
  if (catVars.length === 0) return '暂无分类变量数据'

  // 检查是否有类别数 > 1 的分类变量
  const hasValid = catVars.some(key => {
    const info = summaries[key]
    let vc = {}
    if (info.value_counts && Object.keys(info.value_counts).length > 0) {
      vc = info.value_counts
    } else if (info.top_categories && Object.keys(info.top_categories).length > 0) {
      vc = info.top_categories
    }
    return Object.keys(vc).length > 1
  })

  return hasValid ? '' : '所有分类变量均只有1个类别，无有效分布'
})

const hasCategoricalChartData = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const catVars = Object.keys(summaries).filter(key => {
    const info = summaries[key]
    if (!info.type || !['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)) return false
    const hasValueCounts = info.value_counts && Object.keys(info.value_counts).length > 0
    const hasTopCategories = info.top_categories && Object.keys(info.top_categories).length > 0
    return hasValueCounts || hasTopCategories
  })
  if (catVars.length === 0) return false

  // ✅ 只有存在类别数 > 1 的分类变量时才显示图表
  return catVars.some(key => {
    const info = summaries[key]
    let vc = {}
    if (info.value_counts && Object.keys(info.value_counts).length > 0) {
      vc = info.value_counts
    } else if (info.top_categories && Object.keys(info.top_categories).length > 0) {
      vc = info.top_categories
    }
    return Object.keys(vc).length > 1
  })
})

const categoricalBarOption = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}

  // 找出所有分类变量，并计算它们的类别数
  const catVars = Object.keys(summaries).filter(key => {
    const info = summaries[key]
    if (!info.type || !['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)) return false
    const hasValueCounts = info.value_counts && Object.keys(info.value_counts).length > 0
    const hasTopCategories = info.top_categories && Object.keys(info.top_categories).length > 0
    return hasValueCounts || hasTopCategories
  }).map(key => {
    const info = summaries[key]
    let vc = {}
    if (info.value_counts && Object.keys(info.value_counts).length > 0) {
      vc = info.value_counts
    } else if (info.top_categories && Object.keys(info.top_categories).length > 0) {
      vc = info.top_categories
    }
    return { key, vc, count: Object.keys(vc).length }
  })

  // ✅ 过滤掉类别数 <= 1 的变量
  const validVars = catVars.filter(v => v.count > 1)
  if (validVars.length === 0) return {}

  // ✅ 按类别数从多到少排序，取第一个
  validVars.sort((a, b) => b.count - a.count)
  const best = validVars[0]
  const entries = Object.entries(best.vc).slice(0, 10)

  // 计算总数用于占比
  const total = entries.reduce((s, e) => s + e[1], 0)

  return {
    title: {
      text: `${best.key}（${best.count}个类别）`,
      textStyle: { fontSize: 12, fontWeight: 500, color: '#333' },
      left: 'center',
      top: 0
    },
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        const pct = total > 0 ? (p.value / total * 100).toFixed(1) : 0
        return `<strong>${p.name}</strong><br/>数量：${p.value}<br/>占比：${pct}%`
      }
    },
    grid: { left: '10%', right: '8%', top: '18%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: entries.map(e => String(e[0])),
      axisLabel: { rotate: 30, fontSize: 10, interval: 0 }
    },
    yAxis: { type: 'value', name: '数量' },
    series: [{
      type: 'bar',
      data: entries.map(e => ({ value: e[1], itemStyle: { color: '#409EFF' } })),
      barWidth: '45%',
      label: { show: true, position: 'top', formatter: '{c}', fontSize: 10 }
    }]
  }
})

// ==================== 连续变量分布直方图 ====================
const hasContinuousChartData = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const contVars = Object.keys(summaries).filter(key => summaries[key]?.type === 'continuous')
  return contVars.length > 0
})

const continuousHistOption = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const contVars = Object.keys(summaries).filter(key => summaries[key]?.type === 'continuous')
  if (contVars.length === 0) return {}
  const topVars = contVars.slice(0, 3)
  const colors = ['#409EFF', '#67C23A', '#E6A23C']
  const seriesData = topVars.map((key, idx) => ({
    name: key,
    type: 'bar',
    data: [{ value: summaries[key].mean || 0, itemStyle: { color: colors[idx % colors.length] } }],
    barWidth: '30%',
    label: { show: true, position: 'top', formatter: p => p.value.toFixed(1), fontSize: 10 }
  }))
  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        return params.map(p => `<strong>${p.seriesName}</strong><br/>均值：${p.value.toFixed(2)}`).join('<br/>')
      }
    },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '15%' },
    xAxis: {
      type: 'category',
      data: topVars,
      axisLabel: { fontSize: 11, interval: 0 }
    },
    yAxis: { type: 'value', name: '均值' },
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
    loading.value = false
    return
  }

  loading.value = true
  try {
    const result = await reportApi.get(sessionId)
    console.log('📊 DataOverview - reportData:', result)
    reportData.value = result
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
  max-width: 1400px;
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
  font-size: 32px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
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

@media (max-width: 768px) {
  .charts-row {
    grid-template-columns: 1fr;
  }
  .chart-container {
    height: 200px;
  }
}

/* ===== 其他 ===== */
.section {
  margin-bottom: 30px;
}
.section h4 {
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 16px;
}
.type-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.type-tag {
  font-size: 14px;
  padding: 8px 16px;
}
</style>