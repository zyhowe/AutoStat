<template>
  <div class="data-validation">
    <h2>📋 数据核验</h2>
    <p class="subtitle">检查数据一致性，查看勾稽规则、异常值、缺失值明细及清洗建议</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="validation-content">
      <!-- ===== 表选择器 ===== -->
      <TableSelector
        v-model="currentTable"
        :table-names="tableNames"
        :is-multi-table="isMultiTable"
        @change="onTableChange"
      />

      <!-- ===== 概览柱状图 ===== -->
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

      <el-tabs v-model="activeTab">
        <!-- ===== 勾稽规则 ===== -->
        <el-tab-pane label="勾稽规则" name="rules">
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
                <el-table-column label="样本量" width="100" align="center">
                  <template #default="{ row }">
                    <span class="field-name-link" @click="showRuleSampleData(row)">
                      {{ row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : '--' }}
                    </span>
                  </template>
                </el-table-column>
                <el-table-column label="优先级" width="80" align="center">
                  <template #default="{ row }">
                    <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
                      {{ row.priority }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="violation_count" label="违反数" width="80" align="center">
                  <template #default="{ row }">
                    <span
                      class="field-name-link"
                      @click="showRuleViolations(row)"
                    >
                      {{ row.violation_count }}
                    </span>
                  </template>
                </el-table-column>
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
                <el-table-column label="样本量" width="100" align="center">
                  <template #default="{ row }">
                    <span class="field-name-link" @click="showRuleSampleData(row)">
                      {{ row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : '--' }}
                    </span>
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
                <el-table-column label="样本量" width="100" align="center">
                  <template #default="{ row }">
                    <span class="field-name-link" @click="showRuleSampleData(row)">
                      {{ row.valid_count !== undefined && row.valid_count !== null ? row.valid_count : '--' }}
                    </span>
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
            <el-table-column prop="field" label="字段" width="150" fixed="left">
              <template #default="{ row }">
                <span class="field-name-link" @click="openFieldDetail(row.field)">{{ row.field }}</span>
              </template>
            </el-table-column>
            <el-table-column label="样本量" width="100" align="center">
              <template #default="{ row }">
                <span class="field-name-link" @click="showFieldNonMissing(row.field)">
                  {{ getFieldNonMissingCount(row.field) }}
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="count" label="异常数量" width="120" align="center">
              <template #default="{ row }">
                <span class="field-name-link" @click="showOutlierRows(row.field)">
                  {{ row.count }}
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="percent" label="异常比例" width="120" align="center">
              <template #default="{ row }">
                <span class="field-name-link" @click="showOutlierRows(row.field)">
                  {{ row.percent.toFixed(1) }}%
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="lower_bound" label="下界" width="120" align="center" />
            <el-table-column prop="upper_bound" label="上界" width="120" align="center" />
          </el-table>
        </el-tab-pane>

        <!-- ===== 缺失值 ===== -->
        <el-tab-pane label="缺失值" name="missing">
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
            <el-table-column prop="column" label="字段" width="150" fixed="left">
              <template #default="{ row }">
                <span class="field-name-link" @click="openFieldDetail(row.column)">{{ row.column }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="count" label="缺失数量" width="120" align="center">
              <template #default="{ row }">
                <span class="field-name-link" @click="showMissingRows(row.column)">
                  {{ row.count }}
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="percent" label="缺失比例" width="120" align="center">
              <template #default="{ row }">
                <span class="field-name-link" @click="showMissingRows(row.column)">
                  {{ row.percent.toFixed(1) }}%
                </span>
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
import { ref, computed, onMounted, watch } from 'vue'
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
const activeTab = ref('rules')

// ===== 表选择器状态 =====
const currentTable = ref('merged')
const tableNames = ref([])
const isMultiTable = ref(false)

// ===== 强制刷新 key =====
const overviewKey = ref(0)
const outlierKey = ref(0)
const missingKey = ref(0)
const auditPieKey = ref(0)

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

const currentQuality = computed(() => {
  return currentData.value?.quality_report || {}
})

const currentSummaries = computed(() => {
  return currentData.value?.variable_summaries || {}
})

// ===== 监听 tab 切换 =====
watch(activeTab, (newTab) => {
  if (newTab === 'outliers') {
    setTimeout(() => { outlierKey.value += 1 }, 100)
  } else if (newTab === 'missing') {
    setTimeout(() => { missingKey.value += 1 }, 100)
  } else if (newTab === 'rules') {
    setTimeout(() => { auditPieKey.value += 1 }, 100)
  }
})

function onTableChange() {
  loadData()
}

// ==================== 计算属性 ====================
const auditRules = computed(() => {
  return currentQuality.value?.audit_rules || {}
})

const arithmeticRules = computed(() => {
  const rules = currentQuality.value?.audit_rules?.arithmetic_rules || []
  return rules
})

const functionalRules = computed(() => {
  return currentQuality.value?.audit_rules?.functional_dependencies || []
})

const temporalRules = computed(() => {
  return currentQuality.value?.audit_rules?.temporal_rules || []
})

const auditRulesTotal = computed(() => {
  return arithmeticRules.value.length + functionalRules.value.length + temporalRules.value.length
})

const outlierList = computed(() => {
  const outliers = currentQuality.value?.outliers || {}
  return Object.entries(outliers).map(([field, info]) => ({
    field,
    count: info.count || 0,
    percent: info.percent || 0,
    lower_bound: info.lower_bound,
    upper_bound: info.upper_bound
  }))
})

const missingList = computed(() => {
  return currentQuality.value?.missing || []
})

const duplicateCount = computed(() => {
  return currentQuality.value?.duplicates?.count || 0
})

const duplicateRate = computed(() => {
  return currentQuality.value?.duplicates?.percent || 0
})

const cleaningSuggestions = computed(() => {
  return currentData.value?.cleaning_suggestions || []
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

// ===== 字段详情 =====
function buildFieldData(fieldName) {
  const summary = currentSummaries.value?.[fieldName] || {}
  const variableTypes = currentData.value?.variable_types || {}
  const qualityReport = currentQuality.value

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

// ==================== 辅助函数 ====================

function getFieldNonMissingCount(fieldName) {
  const summary = currentSummaries.value?.[fieldName]
  if (!summary) return 0
  return summary.count || 0
}

// ==================== 数据预览联动 ====================

function showRuleSampleData(rule) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  const fields = rule.fields || []
  if (fields.length === 0) {
    ElMessage.warning('该规则没有关联字段')
    return
  }

  const validCount = rule.valid_count || 0
  const filters = fields.map(f => ({
    field: f,
    condition: 'is_not_null',
    value: true
  }))

  openDataPreview({
    sessionId: sessionId,
    title: `规则「${rule.rule?.substring(0, 40) || '未知规则'}」涉及字段均非空的数据（${validCount} 条）`,
    fields: fields,
    filters: filters
  })
}

function showFieldNonMissing(fieldName) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  openDataPreview({
    sessionId: sessionId,
    title: `「${fieldName}」非空数据（${getFieldNonMissingCount(fieldName)} 条）`,
    filters: [
      { field: fieldName, condition: 'is_not_null', value: true }
    ]
  })
}

function showRuleViolations(rule) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  const fields = rule.fields || []
  if (fields.length === 0) {
    ElMessage.warning('该规则没有关联字段，无法查看')
    return
  }

  const violationCount = rule.violation_count || 0
  const filters = fields.map(f => ({
    field: f,
    condition: 'is_not_null',
    value: true
  }))

  openDataPreview({
    sessionId: sessionId,
    title: `规则「${rule.rule}」涉及字段数据（共 ${fields.length} 个字段，违反 ${violationCount} 条）`,
    fields: fields,
    filters: filters
  })
}

function showOutlierRows(fieldName) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  openDataPreview({
    sessionId: sessionId,
    title: `「${fieldName}」异常值数据`,
    filters: [
      { field: fieldName, condition: 'is_outlier', value: true }
    ]
  })
}

function showMissingRows(fieldName) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  openDataPreview({
    sessionId: sessionId,
    title: `「${fieldName}」为空的数据`,
    filters: [
      { field: fieldName, condition: 'is_null', value: true }
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

    console.log('✅ 数据核验加载完成')
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