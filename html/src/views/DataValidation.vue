<template>
  <div class="data-validation">
    <h2>📋 数据核验</h2>
    <p class="subtitle">检查数据一致性，查看勾稽规则、异常值、缺失值明细及清洗建议</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="validation-content">
      <!-- ===== 概览柱状图（始终显示） ===== -->
      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📊 各维度问题概览</span>
        </div>
        <v-chart
          v-if="hasOverviewData"
          :key="'overview_' + overviewKey"
          :option="overviewBarOption"
          class="chart-container"
          style="height: 200px;"
        />
        <div v-else class="chart-empty">暂无问题数据</div>
      </div>

      <!-- ===== 勾稽规则关系图（饼图） ===== -->
      <div class="chart-card full-width" v-if="hasAuditData">
        <div class="chart-header">
          <span class="chart-title">🔗 勾稽规则类型分布</span>
        </div>
        <v-chart
          :key="'audit_pie_' + auditPieKey"
          :option="auditPieOption"
          class="chart-container"
          style="height: 200px;"
        />
      </div>

      <el-tabs v-model="activeTab">
        <!-- ===== 勾稽规则 ===== -->
        <el-tab-pane label="勾稽规则" name="rules">
          <div v-if="auditRulesTotal === 0" class="empty-tip">
            ✅ 未发现勾稽规则违反，数据一致性良好
          </div>
          <template v-else>
            <div v-if="arithmeticRules.length > 0" class="rule-section">
              <h4>📐 数值关系（{{ arithmeticRules.length }} 条）</h4>
              <el-table :data="arithmeticRules" border size="small" max-height="420">
                <el-table-column prop="rule" label="规则" min-width="200" />
                <el-table-column prop="confidence" label="置信度" width="100" align="center">
                  <template #default="{ row }">
                    {{ (row.confidence * 100).toFixed(1) }}%
                  </template>
                </el-table-column>
                <el-table-column label="优先级" width="80" align="center">
                  <template #default="{ row }">
                    <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
                      {{ row.priority }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="violation_count" label="违反数" width="80" align="center" />
              </el-table>
            </div>

            <div v-if="functionalRules.length > 0" class="rule-section">
              <h4>🏷️ 函数依赖（{{ functionalRules.length }} 条）</h4>
              <el-table :data="functionalRules" border size="small" max-height="420">
                <el-table-column prop="rule" label="规则" min-width="200" />
                <el-table-column prop="confidence" label="置信度" width="100" align="center">
                  <template #default="{ row }">
                    {{ (row.confidence * 100).toFixed(1) }}%
                  </template>
                </el-table-column>
                <el-table-column label="优先级" width="80" align="center">
                  <template #default="{ row }">
                    <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
                      {{ row.priority }}
                    </el-tag>
                  </template>
                </el-table-column>
              </el-table>
            </div>

            <div v-if="temporalRules.length > 0" class="rule-section">
              <h4>📅 时序约束（{{ temporalRules.length }} 条）</h4>
              <el-table :data="temporalRules" border size="small" max-height="420">
                <el-table-column prop="rule" label="规则" min-width="200" />
                <el-table-column prop="confidence" label="置信度" width="100" align="center">
                  <template #default="{ row }">
                    {{ (row.confidence * 100).toFixed(1) }}%
                  </template>
                </el-table-column>
                <el-table-column label="优先级" width="80" align="center">
                  <template #default="{ row }">
                    <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
                      {{ row.priority }}
                    </el-tag>
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </template>
        </el-tab-pane>

        <!-- ===== 异常值 ===== -->
        <el-tab-pane label="异常值" name="outliers">
          <!-- 异常值图表：仅在当前 tab 激活时渲染，并延迟 key 更新 -->
          <div class="chart-card full-width" style="margin-bottom: 16px;">
            <div class="chart-header">
              <span class="chart-title">📊 异常值分布（TOP 5）</span>
            </div>
            <v-chart
              v-if="hasOutlierBoxData && activeTab === 'outliers'"
              :key="'outlier_' + outlierKey"
              :option="boxplotOption"
              class="chart-container"
              style="height: 220px;"
            />
            <div v-else-if="hasOutlierBoxData" class="chart-empty">切换到此标签后显示</div>
            <div v-else class="chart-empty">暂无异常值数据</div>
          </div>

          <div v-if="outlierList.length === 0" class="empty-tip">
            ✅ 未发现异常值
          </div>
          <el-table v-else :data="outlierList" border size="small" max-height="420">
            <el-table-column prop="field" label="字段" width="150" fixed="left" />
            <el-table-column prop="count" label="异常数量" width="120" align="center" />
            <el-table-column prop="percent" label="异常比例" width="120" align="center">
              <template #default="{ row }">
                {{ row.percent.toFixed(1) }}%
              </template>
            </el-table-column>
            <el-table-column prop="lower_bound" label="下界" width="120" align="center" />
            <el-table-column prop="upper_bound" label="上界" width="120" align="center" />
          </el-table>
        </el-tab-pane>

        <!-- ===== 缺失值 ===== -->
        <el-tab-pane label="缺失值" name="missing">
          <!-- 缺失值图表：仅在当前 tab 激活时渲染，并延迟 key 更新 -->
          <div class="chart-card full-width" style="margin-bottom: 16px;">
            <div class="chart-header">
              <span class="chart-title">📊 缺失率 TOP 10</span>
            </div>
            <v-chart
              v-if="hasMissingBarData && activeTab === 'missing'"
              :key="'missing_' + missingKey"
              :option="missingBarOption"
              class="chart-container"
              style="height: 220px;"
            />
            <div v-else-if="hasMissingBarData" class="chart-empty">切换到此标签后显示</div>
            <div v-else class="chart-empty">暂无缺失数据</div>
          </div>

          <div v-if="missingList.length === 0" class="empty-tip">
            ✅ 无缺失值
          </div>
          <el-table v-else :data="missingList" border size="small" max-height="420">
            <el-table-column prop="column" label="字段" width="150" fixed="left" />
            <el-table-column prop="count" label="缺失数量" width="120" align="center" />
            <el-table-column prop="percent" label="缺失比例" width="120" align="center">
              <template #default="{ row }">
                {{ row.percent.toFixed(1) }}%
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <!-- ===== 重复记录 ===== -->
        <el-tab-pane label="重复记录" name="duplicates">
          <div v-if="duplicateCount === 0" class="empty-tip">
            ✅ 无重复记录
          </div>
          <div v-else>
            <el-alert
              :title="`发现 ${duplicateCount} 条重复记录（占比 ${duplicateRate.toFixed(1)}%）`"
              type="warning"
              show-icon
              :closable="false"
            />
          </div>
        </el-tab-pane>

        <!-- ===== 清洗建议 ===== -->
        <el-tab-pane label="🧹 清洗建议" name="cleaning">
          <div v-if="cleaningSuggestions.length === 0" class="empty-tip success">
            ✅ 数据质量良好，无需清洗
          </div>
          <div v-else class="cleaning-scroll">
            <el-timeline>
              <el-timeline-item
                v-for="(suggestion, index) in cleaningSuggestions"
                :key="index"
                :type="index === 0 ? 'primary' : 'info'"
                size="large"
              >
                {{ suggestion }}
              </el-timeline-item>
            </el-timeline>
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
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)
const activeTab = ref('rules')

// ==================== 强制刷新 key ====================
const overviewKey = ref(0)
const outlierKey = ref(0)
const missingKey = ref(0)
const auditPieKey = ref(0)

// ==================== 监听 tab 切换，延迟更新 key 让 DOM 渲染完成 ====================
watch(activeTab, (newTab) => {
  if (newTab === 'outliers') {
    // 延迟 100ms 确保 DOM 已布局
    setTimeout(() => {
      outlierKey.value += 1
    }, 100)
  } else if (newTab === 'missing') {
    setTimeout(() => {
      missingKey.value += 1
    }, 100)
  }
})

// ==================== 计算属性 ====================
const auditRules = computed(() => {
  return reportData.value?.quality_report?.audit_rules || {}
})

const arithmeticRules = computed(() => auditRules.value.arithmetic_rules || [])
const functionalRules = computed(() => auditRules.value.functional_dependencies || [])
const temporalRules = computed(() => auditRules.value.temporal_rules || [])
const auditRulesTotal = computed(() => {
  return arithmeticRules.value.length + functionalRules.value.length + temporalRules.value.length
})

const outlierList = computed(() => {
  const outliers = reportData.value?.quality_report?.outliers || {}
  return Object.entries(outliers).map(([field, info]) => ({
    field,
    count: info.count || 0,
    percent: info.percent || 0,
    lower_bound: info.lower_bound,
    upper_bound: info.upper_bound
  }))
})

const missingList = computed(() => {
  return reportData.value?.quality_report?.missing || []
})

const duplicateCount = computed(() => {
  return reportData.value?.quality_report?.duplicates?.count || 0
})

const duplicateRate = computed(() => {
  return reportData.value?.quality_report?.duplicates?.percent || 0
})

const cleaningSuggestions = computed(() => {
  return reportData.value?.cleaning_suggestions || []
})

// ==================== 概览柱状图 ====================
const hasOverviewData = computed(() => {
  return auditRulesTotal.value > 0 || outlierList.value.length > 0 || missingList.value.length > 0 || duplicateCount.value > 0
})

const overviewBarOption = computed(() => {
  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        return params.map(p => `<strong>${p.name}</strong><br/>${p.seriesName}：${p.value}`).join('<br/>')
      }
    },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '15%' },
    xAxis: {
      type: 'category',
      data: ['勾稽规则', '异常值', '缺失值', '重复记录']
    },
    yAxis: { type: 'value', name: '数量' },
    series: [{
      type: 'bar',
      data: [
        { value: auditRulesTotal.value, itemStyle: { color: '#409EFF' } },
        { value: outlierList.value.length, itemStyle: { color: '#E6A23C' } },
        { value: missingList.value.length, itemStyle: { color: '#F56C6C' } },
        { value: duplicateCount.value, itemStyle: { color: '#67C23A' } }
      ],
      barWidth: '45%',
      label: { show: true, position: 'top', formatter: '{c}', fontSize: 12 }
    }]
  }
})

// ==================== 勾稽规则饼图 ====================
const hasAuditData = computed(() => auditRulesTotal.value > 0)

const auditPieOption = computed(() => {
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      top: 'center',
      itemWidth: 12,
      itemHeight: 12
    },
    series: [{
      type: 'pie',
      radius: ['40%', '65%'],
      center: ['55%', '50%'],
      avoidLabelOverlap: true,
      label: { show: true, formatter: '{b}\n{d}%', fontSize: 11 },
      labelLine: { show: true },
      emphasis: { scale: true },
      data: [
        { value: arithmeticRules.value.length, name: '数值关系', itemStyle: { color: '#409EFF' } },
        { value: functionalRules.value.length, name: '函数依赖', itemStyle: { color: '#67C23A' } },
        { value: temporalRules.value.length, name: '时序约束', itemStyle: { color: '#E6A23C' } }
      ]
    }]
  }
})

// ==================== 异常值柱状图 ====================
const hasOutlierBoxData = computed(() => outlierList.value.length > 0)

const boxplotOption = computed(() => {
  const topOutliers = outlierList.value.slice(0, 5)
  const data = topOutliers.map(item => ({
    name: item.field,
    value: item.count
  }))
  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>异常数量：${p.value}`
      }
    },
    grid: { left: '10%', right: '8%', top: '10%', bottom: '15%' },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLabel: { fontSize: 10, interval: 0 }
    },
    yAxis: { type: 'value', name: '异常数量' },
    series: [{
      type: 'bar',
      data: data.map(d => ({
        value: d.value,
        itemStyle: { color: d.value > 50 ? '#F56C6C' : d.value > 20 ? '#E6A23C' : '#67C23A' }
      })),
      barWidth: '40%',
      label: { show: true, position: 'top', formatter: '{c}', fontSize: 10 }
    }]
  }
})

// ==================== 缺失值条形图 ====================
const hasMissingBarData = computed(() => missingList.value.length > 0)

const missingBarOption = computed(() => {
  const sorted = [...missingList.value].sort((a, b) => (b.percent || 0) - (a.percent || 0))
  const top = sorted.slice(0, 10)
  return {
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const p = params[0]
        return `<strong>${p.name}</strong><br/>缺失率：${p.value}%`
      }
    },
    grid: { left: '12%', right: '8%', top: '10%', bottom: '20%' },
    xAxis: {
      type: 'category',
      data: top.map(m => m.column || '未知'),
      axisLabel: { rotate: 30, fontSize: 10, interval: 0 }
    },
    yAxis: {
      type: 'value',
      max: 100,
      name: '缺失率 (%)',
      nameLocation: 'middle',
      nameGap: 40
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
      label: { show: true, position: 'top', formatter: '{c}%', fontSize: 10 }
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
    console.log('✅ 数据核验加载完成')
    console.log('📊 异常值数据:', result?.quality_report?.outliers)
    console.log('📊 缺失值数据:', result?.quality_report?.missing)
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
.data-validation {
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
  color: #67c23a;
  font-size: 16px;
}
.empty-tip.success {
  color: #67c23a;
}
.rule-section {
  margin-bottom: 24px;
}
.rule-section h4 {
  margin-bottom: 12px;
  color: #555;
  font-size: 14px;
}

/* ===== 图表区域 ===== */
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

.cleaning-scroll {
  max-height: 420px;
  overflow-y: auto;
  padding-right: 8px;
}
.cleaning-scroll::-webkit-scrollbar {
  width: 6px;
}
.cleaning-scroll::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}
.cleaning-scroll::-webkit-scrollbar-track {
  background: #f0f2f6;
  border-radius: 3px;
}
</style>