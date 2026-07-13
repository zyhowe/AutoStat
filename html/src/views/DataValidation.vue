// src/views/DataValidation.vue
// 只修改 parseRuleToExpr 和 showRuleViolations 方法，其余不变
<template>
  <div class="data-validation">
    <h2>📋 数据核验</h2>
    <p class="subtitle">检查数据一致性，查看勾稽规则、异常值、缺失值明细及清洗建议</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="validation-content">
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

const router = useRouter()
const sessionStore = useSessionStore()
const fieldDetailStore = useFieldDetailStore()

const loading = ref(false)
const reportData = ref(null)
const activeTab = ref('rules')

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

// ===== 字段详情 =====
function buildFieldData(fieldName) {
  const summary = reportData.value?.variable_summaries?.[fieldName] || {}
  const varType = reportData.value?.variable_types?.[fieldName]?.type || 'unknown'
  const varTypeDesc = reportData.value?.variable_types?.[fieldName]?.type_desc || typeDisplay[varType] || varType

  const tsDiag = reportData.value?.time_series_diagnostics?.[fieldName] || null
  const outlier = reportData.value?.quality_report?.outliers?.[fieldName] || null
  const missing = reportData.value?.quality_report?.missing?.find(m => m.column === fieldName) || null
  const dupInfo = reportData.value?.quality_report?.duplicates || null

  const correlations = []
  const matrix = reportData.value?.correlations?.matrix || {}
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
  const auditRules = reportData.value?.quality_report?.audit_rules || {}
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
  const modelRecs = reportData.value?.model_recommendations || []
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

/**
 * 解析规则表达式，生成用于筛选违反记录的表达式字符串
 * 增强版：处理更多规则格式
 */
function parseRuleToExpr(rule) {
  const ruleStr = rule.rule || ''
  const fields = rule.fields || []

  // 清理规则字符串：去除多余空格
  const cleanRule = ruleStr.replace(/\s+/g, ' ').trim()

  // 1. 处理 "A = B" 格式（两字段相等）
  if (cleanRule.includes(' = ') && !cleanRule.includes(' + ') && fields.length >= 2) {
    const parts = cleanRule.split(' = ')
    if (parts.length === 2) {
      const left = parts[0].trim()
      const right = parts[1].trim()
      // 检查是否都是字段名（不是数字或常量）
      if (fields.includes(left) && fields.includes(right)) {
        return `abs(${left} - ${right}) > 0.000001`
      }
    }
  }

  // 2. 处理 "A → B" 格式（函数依赖）
  if (cleanRule.includes(' → ')) {
    const parts = cleanRule.split(' → ')
    if (parts.length === 2) {
      const left = parts[0].trim()
      const right = parts[1].trim()
      if (fields.includes(left) && fields.includes(right)) {
        return `${left}.notna() & ${right}.isna()`
      }
    }
  }

  // 3. 处理 "A = B + C" 格式（加法关系，右边有加号）
  if (cleanRule.includes(' = ') && cleanRule.includes(' + ')) {
    const parts = cleanRule.split(' = ')
    if (parts.length === 2) {
      const left = parts[0].trim()
      const right = parts[1].trim()
      // 检查左侧是否是字段，右侧是否包含字段
      if (fields.includes(left)) {
        return `abs(${left} - (${right})) > 0.000001`
      }
      // 检查右侧是否是字段，左侧是否包含字段
      if (fields.includes(right)) {
        // 左侧可能是 "A + B" 形式
        return `abs((${left}) - ${right}) > 0.000001`
      }
    }
  }

  // 4. 处理 "A + B = C" 格式（加法关系，左边有加号）
  if (cleanRule.includes(' + ') && cleanRule.includes(' = ')) {
    const parts = cleanRule.split(' = ')
    if (parts.length === 2) {
      const left = parts[0].trim()
      const right = parts[1].trim()
      if (fields.includes(right)) {
        return `abs((${left}) - ${right}) > 0.000001`
      }
    }
  }

  // 5. 处理 "A = B + C + D" 等复杂加法
  if (cleanRule.includes(' = ') && cleanRule.includes(' + ')) {
    const parts = cleanRule.split(' = ')
    if (parts.length === 2) {
      const left = parts[0].trim()
      const right = parts[1].trim()
      // 尝试提取所有字段
      const leftFields = left.split('+').map(s => s.trim())
      const rightFields = right.split('+').map(s => s.trim())
      // 检查是否所有部分都是字段
      const allLeftAreFields = leftFields.every(f => fields.includes(f))
      const allRightAreFields = rightFields.every(f => fields.includes(f))
      if (allLeftAreFields && allRightAreFields) {
        return `abs((${left}) - (${right})) > 0.000001`
      }
    }
  }

  // 6. 如果规则字符串包含 "="，尝试简单解析
  if (cleanRule.includes('=') && fields.length >= 2) {
    // 尝试按 "=" 分割，取两边所有字段
    const parts = cleanRule.split('=')
    if (parts.length === 2) {
      const leftPart = parts[0].trim()
      const rightPart = parts[1].trim()
      // 提取左右两边包含的字段
      const leftFieldsInRule = fields.filter(f => leftPart.includes(f))
      const rightFieldsInRule = fields.filter(f => rightPart.includes(f))
      if (leftFieldsInRule.length > 0 && rightFieldsInRule.length > 0) {
        // 构建表达式：左边字段的表达式 = 右边字段的表达式
        const leftExpr = leftFieldsInRule.join(' + ')
        const rightExpr = rightFieldsInRule.join(' + ')
        return `abs((${leftExpr}) - (${rightExpr})) > 0.000001`
      }
    }
  }

  // 无法解析，返回 null
  console.warn('⚠️ 无法解析规则表达式:', cleanRule)
  return null
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

  // 尝试解析规则表达式
  const expr = parseRuleToExpr(rule)
  const violationCount = rule.violation_count || 0

  let filters = []
  let title = ''
  let warningMsg = ''

  if (expr) {
    // 精确筛选：使用 expr 条件
    filters = [{ field: 'expr', condition: 'expr', value: expr }]
    title = `违反规则「${rule.rule}」的记录（共 ${fields.length} 个字段，违反 ${violationCount} 条）`
  } else {
    // 回退：所有字段非空
    filters = fields.map(f => ({
      field: f,
      condition: 'is_not_null',
      value: true
    }))
    title = `规则「${rule.rule}」涉及字段数据（共 ${fields.length} 个字段，均非空）`
    warningMsg = '⚠️ 无法精确筛选违反记录，仅显示所有相关字段非空的数据'
  }

  openDataPreview({
    sessionId: sessionId,
    title: title,
    fields: fields,
    filters: filters,
    warning: warningMsg
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