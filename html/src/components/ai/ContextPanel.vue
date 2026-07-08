<template>
  <div class="context-panel">
    <!-- ===== 顶部：上下文选择（多选勾选框） ===== -->
    <div class="context-selector">
      <span class="context-label">📌 上下文</span>
      <el-checkbox-group v-model="selectedContexts" @change="onContextChange">
        <el-checkbox value="json">📊 JSON 结果</el-checkbox>
        <el-checkbox value="upload">📁 上传数据</el-checkbox>
        <el-checkbox value="source">🗄️ 源数据</el-checkbox>
      </el-checkbox-group>
      <el-button
        v-if="isManualOverride"
        size="small"
        text
        @click="resetToAuto"
        title="恢复跟随工具"
      >
        ↺
      </el-button>
    </div>

    <el-divider />

    <!-- ===== 数据快照 ===== -->
    <div class="section">
      <div class="section-header">
        <span class="section-title">📊 数据快照</span>
      </div>
      <div v-if="isLoading" class="loading-skeleton">
        <el-skeleton :rows="3" animated />
      </div>
      <div v-else class="snapshot-grid">
        <div class="snapshot-item">
          <span class="snapshot-value">{{ dataShape.rows || '-' }}</span>
          <span class="snapshot-label">行数</span>
        </div>
        <div class="snapshot-item">
          <span class="snapshot-value">{{ dataShape.columns || '-' }}</span>
          <span class="snapshot-label">列数</span>
        </div>
        <div class="snapshot-item">
          <span class="snapshot-value">{{ variableCounts.continuous || 0 }}</span>
          <span class="snapshot-label">连续</span>
        </div>
        <div class="snapshot-item">
          <span class="snapshot-value">{{ variableCounts.categorical || 0 }}</span>
          <span class="snapshot-label">分类</span>
        </div>
        <div class="snapshot-item">
          <span class="snapshot-value">{{ variableCounts.datetime || 0 }}</span>
          <span class="snapshot-label">日期</span>
        </div>
      </div>
    </div>

    <el-divider />

    <!-- ===== 联动内容 ===== -->
    <div class="section">
      <div class="section-header">
        <span class="section-title">{{ contentTitle }}</span>
      </div>
      <div v-if="isLoading" class="loading-skeleton">
        <el-skeleton :rows="4" animated />
      </div>
      <div v-else class="content-body">
        <!-- 无高亮 -->
        <template v-if="!activeToolId">
          <div v-if="qualityScore !== null" class="quality-display">
            <div class="quality-number">
              <span class="score">{{ qualityScore }}</span>
              <span class="max">/ 100</span>
            </div>
            <div class="quality-grade" :class="gradeClass">
              {{ gradeLabel }}
            </div>
            <el-progress
              :percentage="qualityScore"
              :color="gradeColor"
              :stroke-width="6"
              :show-text="false"
            />
          </div>
          <div v-else class="empty-text">暂无质量评分</div>
          <div class="variable-summary">
            <div class="summary-item">
              <span class="label">连续变量</span>
              <span class="value">{{ variableCounts.continuous }}</span>
            </div>
            <div class="summary-item">
              <span class="label">分类变量</span>
              <span class="value">{{ variableCounts.categorical }}</span>
            </div>
            <div class="summary-item">
              <span class="label">日期变量</span>
              <span class="value">{{ variableCounts.datetime }}</span>
            </div>
          </div>
        </template>

        <!-- 描述字段分布 -->
        <template v-if="activeToolId === 'describe_distribution'">
          <div class="field-list">
            <div v-for="field in fieldList" :key="field.name" class="field-item">
              <span class="field-name">{{ field.name }}</span>
              <span class="field-type">{{ field.type }}</span>
              <span class="field-count">{{ field.count }}</span>
              <span class="field-missing" :class="field.missing_pct > 20 ? 'high' : ''">
                {{ field.missing_pct }}%
              </span>
            </div>
          </div>
        </template>

        <!-- 分析变量相关性 -->
        <template v-if="activeToolId === 'analyze_correlation'">
          <div v-if="correlationPairs.length > 0" class="correlation-list">
            <div
              v-for="pair in correlationPairs.slice(0, 10)"
              :key="pair.var1 + pair.var2"
              class="correlation-item"
            >
              <span class="pair">{{ pair.var1 }} ↔ {{ pair.var2 }}</span>
              <span class="value" :class="Math.abs(pair.value) > 0.7 ? 'strong' : ''">
                r = {{ pair.value }}
              </span>
            </div>
            <div v-if="correlationPairs.length > 10" class="more-hint">
              还有 {{ correlationPairs.length - 10 }} 对
            </div>
          </div>
          <div v-else class="empty-text">暂无相关性数据</div>
        </template>

        <!-- 检测时间序列规律 -->
        <template v-if="activeToolId === 'detect_timeseries'">
          <div v-if="tsFields.length > 0" class="ts-list">
            <div v-for="field in tsFields" :key="field.name" class="ts-item">
              <span class="ts-name">{{ field.name }}</span>
              <span class="ts-status" :class="field.has_autocorrelation ? 'has' : 'none'">
                {{ field.has_autocorrelation ? '✅ 有自相关' : '❌ 无自相关' }}
              </span>
              <span class="ts-stationary" :class="field.is_stationary ? 'stable' : 'unstable'">
                {{ field.is_stationary ? '平稳' : '非平稳' }}
              </span>
            </div>
          </div>
          <div v-else class="empty-text">暂无时间序列数据</div>
        </template>

        <!-- 识别分类特征 -->
        <template v-if="activeToolId === 'identify_categorical'">
          <div v-if="categoricalFields.length > 0" class="cat-list">
            <div v-for="field in categoricalFields" :key="field.name" class="cat-item">
              <span class="cat-name">{{ field.name }}</span>
              <span class="cat-unique">{{ field.n_unique }} 个类别</span>
              <span class="cat-top">众数占比 {{ field.top_pct }}%</span>
            </div>
          </div>
          <div v-else class="empty-text">暂无分类变量数据</div>
        </template>

        <!-- 解读质量评分 -->
        <template v-if="activeToolId === 'interpret_quality'">
          <div v-if="qualityScore !== null" class="quality-detail">
            <div class="dimension-item" v-for="(score, name) in dimensionScores" :key="name">
              <span class="dimension-name">{{ dimensionLabels[name] || name }}</span>
              <el-progress
                :percentage="Math.round(score)"
                :color="getProgressColor(score)"
                :stroke-width="8"
              />
              <span class="dimension-value">{{ Math.round(score) }}%</span>
            </div>
          </div>
          <div v-else class="empty-text">暂无质量评分数据</div>
        </template>

        <!-- 检查异常与缺失 -->
        <template v-if="activeToolId === 'check_outliers'">
          <div class="issue-list">
            <div v-if="outlierFields.length > 0" class="issue-group">
              <span class="issue-label">🚨 异常值字段</span>
              <div v-for="field in outlierFields" :key="field.name" class="issue-item">
                <span>{{ field.name }}</span>
                <span>{{ field.count }} 个 ({{ field.percent }}%)</span>
              </div>
            </div>
            <div v-if="missingFields.length > 0" class="issue-group">
              <span class="issue-label">⚠️ 缺失值字段</span>
              <div v-for="field in missingFields.slice(0, 5)" :key="field.column" class="issue-item">
                <span>{{ field.column }}</span>
                <span>{{ field.count }} ({{ field.percent }}%)</span>
              </div>
            </div>
            <div v-if="outlierFields.length === 0 && missingFields.length === 0" class="empty-text">
              ✅ 未发现异常值和缺失值
            </div>
          </div>
        </template>

        <!-- 验证勾稽规则 -->
        <template v-if="activeToolId === 'validate_rules'">
          <div v-if="auditRulesCount > 0" class="audit-list">
            <div class="audit-summary">
              <span>规则总数: {{ auditRulesCount }}</span>
              <span>违反数: {{ auditViolationsCount }}</span>
              <span>满足率: {{ auditSatisfyRate }}%</span>
            </div>
            <div v-for="rule in auditRules.slice(0, 5)" :key="rule.rule" class="audit-item">
              <span class="rule-text">{{ rule.rule }}</span>
              <span class="rule-confidence">{{ (rule.confidence * 100).toFixed(1) }}%</span>
              <span class="rule-status" :class="rule.violation_count > 0 ? 'violated' : 'ok'">
                {{ rule.violation_count > 0 ? `⚠️ ${rule.violation_count}条违反` : '✅' }}
              </span>
            </div>
          </div>
          <div v-else class="empty-text">未发现勾稽规则</div>
        </template>

        <!-- 检测重复记录 -->
        <template v-if="activeToolId === 'detect_duplicates'">
          <div class="duplicate-display">
            <div class="dup-number">
              <span class="dup-count">{{ duplicateCount }}</span>
              <span class="dup-label">重复记录</span>
            </div>
            <div class="dup-rate">占比 {{ duplicateRate }}%</div>
            <div v-if="duplicateCount === 0" class="empty-text">✅ 无重复记录</div>
          </div>
        </template>

        <!-- 自然语言查数 / 生成SQL -->
        <template v-if="activeToolId === 'natural_query' || activeToolId === 'generate_sql'">
          <div class="data-preview">
            <div class="preview-header">
              <span>数据预览 (前5行)</span>
              <span class="preview-count">{{ dataShape.rows }} 行</span>
            </div>
            <div class="preview-table">
              <div v-if="previewData.length === 0" class="empty-text">暂无数据</div>
              <table v-else>
                <thead>
                  <tr>
                    <th v-for="col in previewColumns" :key="col">{{ col }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, idx) in previewData" :key="idx">
                    <td v-for="col in previewColumns" :key="col">{{ row[col] ?? '-' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>

        <!-- 推荐建模方案 -->
        <template v-if="activeToolId === 'recommend_model'">
          <div v-if="modelRecommendations.length > 0" class="model-list">
            <div v-for="rec in modelRecommendations" :key="rec.task_type" class="model-item">
              <span class="model-task">{{ rec.task_type }}</span>
              <span class="model-name">{{ rec.ml }}</span>
              <span class="model-target">目标: {{ rec.target_column || '-' }}</span>
            </div>
          </div>
          <div v-else class="empty-text">暂无建模建议</div>
        </template>

        <!-- 执行预测 -->
        <template v-if="activeToolId === 'execute_predict'">
          <div class="predict-info">
            <div v-if="trainedModels.length > 0" class="model-list">
              <div v-for="model in trainedModels" :key="model.model_key" class="model-item">
                <span class="model-name">{{ model.user_model_name || model.model_key }}</span>
                <span class="model-target">目标: {{ model.target_column || '-' }}</span>
                <span class="model-features">{{ (model.features || []).join('、') }}</span>
              </div>
            </div>
            <div v-else class="empty-text">暂无已训练模型</div>
          </div>
        </template>

        <!-- 生成核心结论 -->
        <template v-if="activeToolId === 'generate_conclusions'">
          <div v-if="conclusions.length > 0" class="conclusion-list">
            <div v-for="(item, idx) in conclusions" :key="idx" class="conclusion-item">
              <span class="conclusion-icon">{{ item.icon || '📌' }}</span>
              <span class="conclusion-text">{{ item.title }}</span>
            </div>
          </div>
          <div v-else class="empty-text">暂无核心结论</div>
        </template>

        <!-- 提炼业务洞察 -->
        <template v-if="activeToolId === 'extract_insights'">
          <div v-if="insightsList.length > 0" class="insight-list">
            <div v-for="(item, idx) in insightsList" :key="idx" class="insight-item">
              <span class="insight-icon">💡</span>
              <span class="insight-text">{{ item }}</span>
            </div>
          </div>
          <div v-else class="empty-text">暂无业务洞察</div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  dataShape: { type: Object, default: () => ({ rows: 0, columns: 0 }) },
  variableTypes: { type: Object, default: () => ({}) },
  qualityScore: { type: Number, default: null },
  dimensionScores: { type: Object, default: () => ({}) },
  activeToolId: { type: String, default: '' },
  fieldList: { type: Array, default: () => [] },
  correlationPairs: { type: Array, default: () => [] },
  tsFields: { type: Array, default: () => [] },
  categoricalFields: { type: Array, default: () => [] },
  outlierFields: { type: Array, default: () => [] },
  missingFields: { type: Array, default: () => [] },
  auditRules: { type: Array, default: () => [] },
  auditRulesCount: { type: Number, default: 0 },
  auditViolationsCount: { type: Number, default: 0 },
  auditSatisfyRate: { type: Number, default: 0 },
  duplicateCount: { type: Number, default: 0 },
  duplicateRate: { type: Number, default: 0 },
  previewData: { type: Array, default: () => [] },
  previewColumns: { type: Array, default: () => [] },
  modelRecommendations: { type: Array, default: () => [] },
  trainedModels: { type: Array, default: () => [] },
  conclusions: { type: Array, default: () => [] },
  insightsList: { type: Array, default: () => [] },
  isLoading: { type: Boolean, default: false }
})

const emit = defineEmits(['context-change'])

// ===== 状态 =====
const selectedContexts = ref(['json'])
const isManualOverride = ref(false)

// ===== 工具 → 默认上下文映射 =====
const toolContextMap = {
  'describe_distribution': ['json'],
  'analyze_correlation': ['json'],
  'detect_timeseries': ['json'],
  'identify_categorical': ['json'],
  'interpret_quality': ['json'],
  'check_outliers': ['json'],
  'validate_rules': ['json'],
  'detect_duplicates': ['json'],
  'recommend_model': ['json'],
  'execute_predict': ['json'],
  'generate_conclusions': ['json'],
  'extract_insights': ['json'],
  'natural_query': ['upload'],
  'generate_sql': ['upload']
}

// ===== 计算属性 =====
const variableCounts = computed(() => {
  const types = props.variableTypes || {}
  let continuous = 0, categorical = 0, datetime = 0, identifier = 0, text = 0
  Object.values(types).forEach(info => {
    const type = info.type || info
    if (type === 'continuous') continuous++
    else if (type === 'categorical' || type === 'categorical_numeric' || type === 'ordinal') categorical++
    else if (type === 'datetime') datetime++
    else if (type === 'identifier') identifier++
    else if (type === 'text') text++
  })
  return { continuous, categorical, datetime, identifier, text }
})

const contentTitle = computed(() => {
  const titles = {
    'describe_distribution': '📋 字段列表',
    'analyze_correlation': '🔗 相关性分析',
    'detect_timeseries': '📈 时间序列',
    'identify_categorical': '🏷️ 分类特征',
    'interpret_quality': '⭐ 质量详情',
    'check_outliers': '🚨 异常与缺失',
    'validate_rules': '🔗 勾稽规则',
    'detect_duplicates': '📋 重复记录',
    'natural_query': '🔍 数据预览',
    'generate_sql': '📝 表结构',
    'recommend_model': '🤖 模型推荐',
    'execute_predict': '🔮 已训练模型',
    'generate_conclusions': '📋 核心结论',
    'extract_insights': '💡 业务洞察'
  }
  return titles[props.activeToolId] || '📊 数据概览'
})

const gradeLabel = computed(() => {
  const score = props.qualityScore
  if (score === null || score === undefined) return '未知'
  if (score >= 90) return '优秀'
  if (score >= 80) return '良好'
  if (score >= 70) return '一般'
  if (score >= 60) return '较差'
  return '差'
})

const gradeClass = computed(() => {
  const score = props.qualityScore
  if (score === null || score === undefined) return ''
  if (score >= 80) return 'grade-good'
  if (score >= 60) return 'grade-warn'
  return 'grade-bad'
})

const gradeColor = computed(() => {
  const score = props.qualityScore
  if (score === null || score === undefined) return '#909399'
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
})

const dimensionLabels = {
  completeness: '完整性',
  accuracy: '准确性',
  consistency: '一致性',
  uniqueness: '唯一性',
  timeliness: '及时性'
}

// ===== 方法 =====
function getProgressColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}

function getDefaultContexts(toolId) {
  return toolContextMap[toolId] || ['json']
}

function onContextChange(value) {
  isManualOverride.value = true
  emit('context-change', {
    contexts: value,
    isManual: isManualOverride.value
  })
}

function resetToAuto() {
  isManualOverride.value = false
  const defaults = getDefaultContexts(props.activeToolId)
  selectedContexts.value = [...defaults]
  emit('context-change', {
    contexts: defaults,
    isManual: false
  })
}

// ===== 监听工具变化 =====
watch(() => props.activeToolId, (newId) => {
  if (!isManualOverride.value && newId) {
    const defaults = getDefaultContexts(newId)
    selectedContexts.value = [...defaults]
    emit('context-change', {
      contexts: defaults,
      isManual: false
    })
  }
}, { immediate: true })

defineExpose({
  resetToAuto,
  getCurrentContexts: () => selectedContexts.value,
  isManual: () => isManualOverride.value
})
</script>

<style scoped>
.context-panel {
  padding: 12px 16px;
  height: 100%;
  overflow-y: auto;
  background: #fafafa;
  border-left: 1px solid #e4e7ed;
}

.context-selector {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 0 8px 0;
  flex-wrap: wrap;
}

.context-label {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
  white-space: nowrap;
}

.context-selector .el-checkbox-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.context-selector .el-checkbox {
  margin-right: 0;
  font-size: 12px;
}

.section {
  margin-bottom: 4px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.section-title {
  font-size: 13px;
  font-weight: 600;
  color: #2c3e50;
}

.loading-skeleton {
  padding: 8px 0;
}

.empty-text {
  padding: 16px 0;
  text-align: center;
  color: #bbb;
  font-size: 13px;
}

/* ===== 数据快照 ===== */
.snapshot-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.snapshot-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 4px;
  background: white;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
}

.snapshot-value {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.snapshot-label {
  font-size: 10px;
  color: #909399;
}

/* ===== 质量评分 ===== */
.quality-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0;
  background: white;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.quality-number {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.quality-number .score {
  font-size: 28px;
  font-weight: bold;
  color: #2c3e50;
}

.quality-number .max {
  font-size: 14px;
  color: #bbb;
}

.quality-grade {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 4px;
}

.quality-grade.grade-good { color: #67c23a; }
.quality-grade.grade-warn { color: #e6a23c; }
.quality-grade.grade-bad { color: #f56c6c; }

/* ===== 变量摘要 ===== */
.variable-summary {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-top: 8px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 10px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f0f0f0;
  font-size: 12px;
}

.summary-item .label { color: #909399; }
.summary-item .value { font-weight: 500; color: #2c3e50; }

/* ===== 字段列表 ===== */
.field-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 300px;
  overflow-y: auto;
}

.field-item {
  display: flex;
  justify-content: space-between;
  padding: 3px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
  align-items: center;
}

.field-name { font-weight: 500; color: #2c3e50; flex: 1; }
.field-type { color: #909399; margin: 0 8px; font-size: 11px; }
.field-count { color: #666; margin: 0 8px; }
.field-missing { color: #67c23a; font-size: 11px; min-width: 40px; text-align: right; }
.field-missing.high { color: #f56c6c; }

/* ===== 相关性 ===== */
.correlation-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 300px;
  overflow-y: auto;
}

.correlation-item {
  display: flex;
  justify-content: space-between;
  padding: 3px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
}

.correlation-item .pair { color: #2c3e50; }
.correlation-item .value { font-weight: 500; color: #909399; }
.correlation-item .value.strong { color: #409eff; }
.more-hint { text-align: center; font-size: 11px; color: #bbb; padding: 4px 0; }

/* ===== 时间序列 ===== */
.ts-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 300px;
  overflow-y: auto;
}

.ts-item {
  display: flex;
  justify-content: space-between;
  padding: 3px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
  align-items: center;
}

.ts-name { font-weight: 500; color: #2c3e50; flex: 1; }
.ts-status { font-size: 11px; margin: 0 8px; }
.ts-status.has { color: #67c23a; }
.ts-status.none { color: #909399; }
.ts-stationary { font-size: 11px; }
.ts-stationary.stable { color: #67c23a; }
.ts-stationary.unstable { color: #e6a23c; }

/* ===== 分类特征 ===== */
.cat-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 300px;
  overflow-y: auto;
}

.cat-item {
  display: flex;
  justify-content: space-between;
  padding: 3px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
  align-items: center;
}

.cat-name { font-weight: 500; color: #2c3e50; flex: 1; }
.cat-unique { color: #666; margin: 0 8px; }
.cat-top { color: #909399; font-size: 11px; }

/* ===== 质量详情 ===== */
.quality-detail { display: flex; flex-direction: column; gap: 6px; }
.dimension-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.dimension-name { font-size: 12px; color: #666; min-width: 50px; }
.dimension-item .el-progress { flex: 1; }
.dimension-value { font-size: 12px; font-weight: 500; color: #2c3e50; min-width: 40px; text-align: right; }

/* ===== 异常与缺失 ===== */
.issue-list { display: flex; flex-direction: column; gap: 8px; }
.issue-group { display: flex; flex-direction: column; gap: 2px; }
.issue-label { font-size: 12px; font-weight: 500; color: #666; padding: 2px 0; }
.issue-item {
  display: flex;
  justify-content: space-between;
  padding: 2px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
}

/* ===== 勾稽规则 ===== */
.audit-list { display: flex; flex-direction: column; gap: 4px; }
.audit-summary {
  display: flex;
  gap: 12px;
  padding: 6px 10px;
  background: #ecf5ff;
  border-radius: 6px;
  font-size: 12px;
  color: #2c3e50;
}
.audit-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #f5f5f5;
  font-size: 12px;
  align-items: center;
}
.rule-text { flex: 1; font-family: monospace; font-size: 11px; color: #2c3e50; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rule-confidence { color: #909399; margin: 0 8px; font-size: 11px; }
.rule-status { font-size: 11px; }
.rule-status.violated { color: #f56c6c; }
.rule-status.ok { color: #67c23a; }

/* ===== 重复记录 ===== */
.duplicate-display { text-align: center; padding: 12px 0; }
.dup-number { display: flex; align-items: baseline; justify-content: center; gap: 8px; }
.dup-count { font-size: 32px; font-weight: bold; color: #2c3e50; }
.dup-label { font-size: 14px; color: #909399; }
.dup-rate { font-size: 13px; color: #666; margin-top: 4px; }

/* ===== 数据预览 ===== */
.data-preview {
  background: white;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
  overflow: hidden;
}
.preview-header {
  display: flex;
  justify-content: space-between;
  padding: 6px 10px;
  background: #f5f7fa;
  font-size: 12px;
  color: #666;
  border-bottom: 1px solid #f0f0f0;
}
.preview-count { color: #909399; }
.preview-table { overflow-x: auto; padding: 4px; }
.preview-table table { width: 100%; border-collapse: collapse; font-size: 11px; }
.preview-table th, .preview-table td {
  padding: 4px 8px;
  border: 1px solid #f0f0f0;
  text-align: left;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.preview-table th { background: #fafafa; font-weight: 500; color: #666; }

/* ===== 模型列表 ===== */
.model-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 300px;
  overflow-y: auto;
}
.model-item {
  display: flex;
  flex-direction: column;
  padding: 6px 10px;
  background: white;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
  gap: 2px;
}
.model-task { font-size: 12px; font-weight: 500; color: #2c3e50; }
.model-name { font-size: 12px; color: #409eff; }
.model-target { font-size: 11px; color: #909399; }
.model-features { font-size: 11px; color: #666; }

/* ===== 核心结论 ===== */
.conclusion-list { display: flex; flex-direction: column; gap: 4px; }
.conclusion-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: white;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
}
.conclusion-icon { font-size: 16px; }
.conclusion-text { font-size: 13px; color: #2c3e50; }

/* ===== 洞察列表 ===== */
.insight-list { display: flex; flex-direction: column; gap: 4px; }
.insight-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: white;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
}
.insight-icon { font-size: 14px; }
.insight-text { font-size: 13px; color: #2c3e50; }

.context-panel::-webkit-scrollbar { width: 4px; }
.context-panel::-webkit-scrollbar-thumb { background: #d0d4dc; border-radius: 2px; }
.context-panel::-webkit-scrollbar-track { background: transparent; }
</style>