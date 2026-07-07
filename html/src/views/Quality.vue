<template>
  <div class="quality-page">
    <h2>📊 数据质量看板</h2>
    <p class="subtitle">四维质量评分，全面了解数据健康状况</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="qualityData" class="quality-content">
      <!-- ===== 图表区域 ===== -->
      <div class="charts-row">
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

        <!-- 雷达图 -->
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 四维质量得分</span>
          </div>
          <v-chart v-if="hasRadarData" :option="radarOption" class="chart-container" />
          <div v-else class="chart-empty">暂无维度数据</div>
        </div>
      </div>

      <!-- ===== 四维评分卡片（进度条） ===== -->
      <div class="dimensions">
        <el-card v-for="(score, name) in filteredDimensions" :key="name" class="dimension-card" shadow="hover">
          <div class="dimension-name">{{ getDimensionLabel(name) }}</div>
          <el-progress
            :percentage="Math.round(score)"
            :color="getProgressColor(score)"
            :stroke-width="12"
          />
          <div class="dimension-value">{{ Math.round(score) }}%</div>
        </el-card>
      </div>

      <!-- ===== 问题清单 ===== -->
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
          <el-table-column prop="field" label="字段" width="150" />
          <el-table-column prop="message" label="问题描述" min-width="200" />
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
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(true)
const qualityData = ref(null)
const issues = ref([])

const gaugeKey = ref(0)

watch(() => qualityData.value?.overall_score, (newVal) => {
  if (newVal !== undefined && newVal !== null) {
    gaugeKey.value += 1
  }
})

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
    const result = await reportApi.getQuality(sessionId)
    console.log('📊 Quality 数据:', result)
    qualityData.value = result

    const allIssues = []
    result?.alerts?.forEach(alert => {
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

  } catch (err) {
    ElMessage.error('加载质量报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
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

function getProgressColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}

// ==================== 过滤 timeliness ====================
const filteredDimensions = computed(() => {
  const dims = qualityData.value?.dimensions || {}
  const filtered = {}
  Object.keys(dims).forEach(k => {
    if (k !== 'timeliness') {
      filtered[k] = dims[k]
    }
  })
  return filtered
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

// ==================== 雷达图 ====================
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

function goTo(routeName) {
  router.push(`/${routeName}`)
}
</script>

<style scoped>
.quality-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  padding-bottom: 40px;
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

/* ===== 维度评分卡片 ===== */
.dimensions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 30px;
}
.dimension-card {
  padding: 16px;
}
.dimension-name {
  font-weight: 500;
  margin-bottom: 8px;
  color: #2c3e50;
}
.dimension-value {
  text-align: right;
  margin-top: 4px;
  font-weight: 500;
  font-size: 14px;
  color: #666;
}

/* ===== 问题清单 ===== */
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
</style>