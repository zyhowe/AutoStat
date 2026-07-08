<template>
  <div class="ai-assistant">
    <div class="workbench-body">
      <div class="tool-panel-wrapper">
        <ToolPanel
          ref="toolPanelRef"
          :scene="currentScene"
          :session-id="sessionStore.currentSessionId"
          :personalized-questions="personalizedQuestions"
          @question-click="handleQuestionClick"
          @refresh-questions="loadRecommendedQuestions"
        />
      </div>

      <div class="chat-wrapper">
        <ChatArea
          ref="chatAreaRef"
          :messages="chatStore.messages"
          :is-streaming="chatStore.isStreaming"
          :streaming-content="chatStore.streamingContent"
          :pending-tool="pendingTool"
          @send="handleSend"
          @clear="handleClear"
          @tool-clear="pendingTool = null"
        />
      </div>

      <div class="context-panel-wrapper">
        <ContextPanel
          ref="contextPanelRef"
          :data-shape="dataShape"
          :variable-types="variableTypes"
          :quality-score="qualityScore"
          :dimension-scores="dimensionScores"
          :active-tool-id="activeToolId"
          :field-list="fieldList"
          :correlation-pairs="correlationPairs"
          :ts-fields="tsFields"
          :categorical-fields="categoricalFields"
          :outlier-fields="outlierFields"
          :missing-fields="missingFields"
          :audit-rules="auditRules"
          :audit-rules-count="auditRulesCount"
          :audit-violations-count="auditViolationsCount"
          :audit-satisfy-rate="auditSatisfyRate"
          :duplicate-count="duplicateCount"
          :duplicate-rate="duplicateRate"
          :preview-data="previewData"
          :preview-columns="previewColumns"
          :model-recommendations="modelRecommendations"
          :trained-models="trainedModels"
          :conclusions="conclusions"
          :insights-list="insightsList"
          :is-loading="loadingContext"
          @context-change="handleContextChange"
        />
      </div>
    </div>

    <div class="footer-toolbar">
      <el-button size="small" text @click="handleClear">🗑️ 清空对话</el-button>
      <el-button size="small" text @click="handleExport">📥 导出对话</el-button>
      <el-tag v-if="chatStore.messages.length > 0" size="small" type="info">
        {{ chatStore.messages.length }} 条消息
      </el-tag>
      <el-tag v-if="currentContexts.length > 0 && currentContexts.length < 3" size="small" type="warning">
        上下文: {{ getContextLabels().join('、') }}
      </el-tag>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useChatStore } from '../stores/chat'
import { reportApi } from '../api/report'
import api from '../api'
import ToolPanel from '../components/ai/ToolPanel.vue'
import ChatArea from '../components/ai/ChatArea.vue'
import ContextPanel from '../components/ai/ContextPanel.vue'

const router = useRouter()
const route = useRoute()
const sessionStore = useSessionStore()
const chatStore = useChatStore()

// ===== Refs =====
const toolPanelRef = ref(null)
const chatAreaRef = ref(null)
const contextPanelRef = ref(null)
const pendingTool = ref(null)
const loadingContext = ref(false)
const reportData = ref(null)
const personalizedQuestions = ref({})
const activeToolId = ref('')

const currentContexts = ref(['json'])
const isManualOverride = ref(false)

// ===== 场景映射 =====
const sceneMap = {
  '/report-summary': 'report_summary',
  '/data-overview': 'data_overview',
  '/quality': 'quality',
  '/data-validation': 'data_validation',
  '/pattern-discovery': 'pattern_discovery',
  '/models': 'smart_prediction'
}

// ===== 场景前缀映射 =====
const scenePrefixMap = {
  'report_summary': '基于当前数据分析报告，',
  'data_overview': '基于当前数据概览，',
  'quality': '基于当前数据质量评分，',
  'data_validation': '基于当前数据核验结果，',
  'pattern_discovery': '基于当前数据规律发现结果，',
  'smart_prediction': '基于当前智能预测分析，',
  'general': '基于当前数据，'
}

// ===== 上下文前缀映射 =====
const contextPrefixMap = {
  'json': 'JSON 分析结果',
  'upload': '上传的原始数据',
  'source': '源数据'
}

// ===== 共性问题增强映射 =====
const questionEnhanceMap = {
  '总结数据的核心特征和整体分布': '请从整体上总结数据的核心特征和分布规律，包括数据规模、变量类型、关键统计指标等',
  '数据的基本规模和各变量类型分布是怎样的？': '请描述数据的基本规模和变量类型分布情况',
  '提炼数据中的关键结论': '请提炼出3-5个最关键的数据结论和发现',
  '数据中最值得关注的 3 个核心发现是什么？': '请提取3个最值得关注的核心发现，并说明其业务意义',
  '给出可执行的业务洞察和建议': '请给出可执行的业务洞察和具体建议',
  '基于当前数据，下一步应该关注什么？': '请指出下一步应该关注的重点方向',
  '各字段的数据分布和统计特征是怎样的？': '请详细分析各字段的数据分布和统计特征，包括连续变量的均值/中位数/标准差，分类变量的类别分布等',
  '数值变量的均值、中位数、标准差如何？': '请列出各数值变量的均值、中位数和标准差，并说明分布形态',
  '分类变量的典型模式和失衡情况如何？': '请分析分类变量的典型模式和失衡情况',
  '连续变量的集中趋势和离散程度如何？': '请分析连续变量的集中趋势和离散程度',
  '日期字段的时间范围和分布特征是什么？': '请分析日期字段的时间范围和分布特征',
  '哪些字段缺失率较高？建议如何处理？': '请列出缺失率较高的字段并给出处理建议',
  '整体缺失率是多少？各字段缺失率排名如何？': '请计算整体缺失率，并按缺失率从高到低排名各字段',
  '综合质量评分如何解读？处于什么水平？': '请解读综合质量评分的含义和所处水平',
  '质量评分的等级和置信度如何？': '请说明质量评分的等级和置信度',
  '完整性得分如何？哪些字段缺失最多？': '请分析完整性得分并列出缺失最多的字段',
  '准确性得分如何？哪些字段异常值最多？': '请分析准确性得分并列出异常值最多的字段',
  '一致性得分如何？勾稽规则满足率是多少？': '请分析一致性得分并计算勾稽规则满足率',
  '当前数据有哪些勾稽规则？哪些被违反？': '请列出所有勾稽规则并指出被违反的规则',
  '勾稽规则的置信度分布如何？': '请分析各规则的置信度分布情况',
  '哪些字段存在异常值？最严重的是哪个？': '请列出存在异常值的字段，并指出最严重的那个',
  '数据中是否存在重复记录？': '请检查是否存在重复记录，如果存在说明数量和占比',
  '给出数据清洗建议': '请给出具体的数据清洗建议',
  '各变量之间的相关性如何？有哪些强相关关系？': '请分析各变量之间的相关性，并列出强相关关系',
  '数值变量之间的相关系数矩阵分析': '请分析各数值变量之间的相关系数矩阵',
  '哪些指标适合做时间序列分析？有什么规律？': '请分析哪些指标适合做时间序列分析并指出其规律',
  '数据存在什么趋势或周期性变化？': '请说明数据存在什么趋势或周期性变化',
  '根据数据特征推荐适合的建模方案': '请推荐适合的建模方案并说明推荐理由',
  '推荐的目标变量有哪些？为什么？': '请推荐适合作为预测目标变量的字段并说明理由',
  '推荐哪些字段作为预测特征？': '请推荐适合作为预测特征的字段',
  '对主要指标进行未来趋势预测': '请对主要指标进行未来趋势预测',
  '分析各变量之间的关联关系': '请分析各变量之间的关联关系',
  '评估时间序列分析的条件': '请评估是否满足时间序列分析的条件',
  '推荐适合的分析方法和建模方案': '请推荐适合的分析方法和建模方案',
  '检查数据质量，指出存在的问题': '请检查数据质量并指出存在的问题',
  '总结关键发现并给出下一步建议': '请总结关键发现并给出下一步建议',
  '查询指定时间段内的数据': '请根据日期字段，查询指定时间段内的数据',
  '按条件筛选数据': '请根据条件筛选数据',
  '查询记录数最多的前N条数据': '请查询记录数最多的前N条数据',
  '生成查询指定时间段数据的SQL语句': '请生成查询指定时间段数据的SQL语句',
  '生成按分类字段分组的统计SQL语句': '请生成按分类字段分组的统计SQL语句',
  '生成两表关联查询的SQL语句': '请生成两表关联查询的SQL语句'
}

// ===== 计算属性 =====
const currentScene = computed(() => {
  const path = route.path
  return sceneMap[path] || 'general'
})

const dataShape = computed(() => {
  const session = sessionStore.currentSession
  return session?.data_shape || { rows: 0, columns: 0 }
})

const variableTypes = computed(() => {
  const session = sessionStore.currentSession
  return session?.variable_types || {}
})

const qualityScore = computed(() => {
  return reportData.value?.overall_score ?? null
})

const dimensionScores = computed(() => {
  return reportData.value?.dimensions || {}
})

// ===== 各种数据 =====
const fieldList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type: info.type_desc || info.type,
    count: info.count || 0,
    missing_pct: info.missing_pct || 0
  }))
})

const correlationPairs = computed(() => {
  return reportData.value?.correlations?.high_correlations || []
})

const tsFields = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  return Object.entries(diag).map(([name, info]) => ({
    name,
    has_autocorrelation: info.has_autocorrelation || false,
    is_stationary: info.is_stationary || false
  }))
})

const categoricalFields = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const result = []
  for (const [name, info] of Object.entries(summaries)) {
    if (info.type === 'categorical' || info.type === 'categorical_numeric' || info.type === 'ordinal') {
      result.push({
        name,
        n_unique: info.n_unique || 0,
        top_pct: info.mode_pct || 0
      })
    }
  }
  return result
})

const outlierFields = computed(() => {
  const outliers = reportData.value?.outliers || {}
  return Object.entries(outliers).map(([name, info]) => ({
    name,
    count: info.count || 0,
    percent: info.percent || 0
  }))
})

const missingFields = computed(() => {
  return reportData.value?.missing || []
})

const auditRules = computed(() => {
  const rules = reportData.value?.audit_rules || {}
  return [
    ...(rules.arithmetic_rules || []),
    ...(rules.functional_dependencies || []),
    ...(rules.temporal_rules || [])
  ]
})

const auditRulesCount = computed(() => auditRules.value.length)
const auditViolationsCount = computed(() => {
  return auditRules.value.reduce((sum, r) => sum + (r.violation_count || 0), 0)
})
const auditSatisfyRate = computed(() => {
  if (auditRulesCount.value === 0) return 100
  const total = auditRules.value.reduce((sum, r) => sum + (r.valid_rows || 0), 0)
  const satisfied = auditRules.value.reduce((sum, r) => sum + (r.satisfied_rows || 0), 0)
  return total > 0 ? Math.round((satisfied / total) * 100) : 0
})

const duplicateCount = computed(() => {
  return reportData.value?.duplicates?.count || 0
})

const duplicateRate = computed(() => {
  return reportData.value?.duplicates?.percent || 0
})

const previewData = computed(() => {
  return sessionStore.previewData || []
})

const previewColumns = computed(() => {
  return Object.keys(variableTypes.value)
})

const modelRecommendations = computed(() => {
  return reportData.value?.model_recommendations || []
})

const trainedModels = computed(() => {
  return reportData.value?.trained_models || []
})

const conclusions = computed(() => {
  return reportData.value?.summary || []
})

const insightsList = computed(() => {
  return reportData.value?.insights?.findings || []
})

// ===== 核心：通过 dataKey 从 recommended_questions JSON 中提取数据 =====
function getDataByKey(dataKey) {
  if (!dataKey) return null

  const keys = dataKey.split('.')
  let data = personalizedQuestions.value

  for (const key of keys) {
    if (data && typeof data === 'object' && key in data) {
      data = data[key]
    } else {
      return null
    }
  }

  return data
}

// ===== 构建上下文数据 =====
function getContextLabels() {
  return currentContexts.value.map(c => contextPrefixMap[c] || c)
}

function buildDefaultContext() {
  const session = sessionStore.currentSession
  const sourceTable = session?.source_name || '未知表名'

  return {
    source_table: sourceTable,
    rows: dataShape.value.rows,
    columns: dataShape.value.columns,
    variableTypes: Object.keys(variableTypes.value).reduce((acc, key) => {
      acc[key] = variableTypes.value[key]?.type || 'unknown'
      return acc
    }, {}),
    qualityScore: qualityScore.value,
    dimensionScores: dimensionScores.value,
    fieldSummary: fieldList.value.slice(0, 10).map(f => ({
      name: f.name,
      type: f.type,
      missing_pct: f.missing_pct
    })),
    topCorrelations: correlationPairs.value.slice(0, 5),
    topOutliers: outlierFields.value.slice(0, 3),
    topMissing: missingFields.value.slice(0, 3),
    auditRulesCount: auditRulesCount.value,
    topModelRecommendations: modelRecommendations.value.slice(0, 2).map(r => ({
      task_type: r.task_type,
      target: r.target_column,
      ml: r.ml
    }))
  }
}

function buildDetailedContext(question) {
  const fieldNames = question.match(/「([^」]+)」/g)?.map(s => s.replace(/[「」]/g, '')) || []
  const context = buildDefaultContext()
  const summaries = reportData.value?.variable_summaries || {}

  if (fieldNames.length > 0) {
    context.detailedFields = {}
    fieldNames.forEach(field => {
      const info = summaries[field]
      if (info) {
        context.detailedFields[field] = {
          type: info.type,
          count: info.count,
          missing: info.missing,
          missing_pct: info.missing_pct,
          mean: info.mean,
          median: info.median,
          std: info.std,
          min: info.min,
          max: info.max,
          skew: info.skew,
          n_unique: info.n_unique,
          mode: info.mode,
          mode_pct: info.mode_pct,
          min_date: info.min_date,
          max_date: info.max_date
        }
      }
    })
  }

  return context
}

function buildEnhancedQuestion(originalText, scene, contexts) {
  const activeContexts = contexts.length > 0 ? contexts : ['json']
  const scenePre = scenePrefixMap[scene] || scenePrefixMap['general']
  const contextStr = activeContexts.map(c => contextPrefixMap[c] || c).join('、')

  const isPersonalized = /「[^」]+」/.test(originalText)

  if (isPersonalized) {
    return `${scenePre}请结合数据源（${contextStr}）中的实际数据，回答：${originalText}`
  } else {
    const enhanced = questionEnhanceMap[originalText]
    if (enhanced) {
      return `${scenePre}${enhanced}（数据来源：${contextStr}）`
    }
    return `${scenePre}请基于数据源（${contextStr}）分析，回答：${originalText}`
  }
}

// ===== 处理方法 =====

function handleQuestionClick(payload) {
  const { text, dataKey } = payload
  const scene = currentScene.value
  const contexts = currentContexts.value

  // 1. 构建增强问题
  const enhanced = buildEnhancedQuestion(text, scene, contexts)

  // 2. 提取 dataKey 对应的数据
  let contextData = buildDefaultContext()
  if (dataKey) {
    const keyData = getDataByKey(dataKey)
    if (keyData) {
      contextData = {
        ...contextData,
        recommendedData: keyData,
        recommendedDataKey: dataKey
      }
    }
  }

  // 3. 如果是个性化问题，提取详细字段信息
  if (/「[^」]+」/.test(text)) {
    const detailed = buildDetailedContext(text)
    contextData = { ...contextData, ...detailed }
  }

  handleSend(enhanced, contextData)
}

function handleSend(text, contextData = null) {
  const message = text || pendingTool.value?.prompt
  if (!message) return

  const defaultContext = contextData || buildDefaultContext()

  const payload = {
    question: message,
    context_data: defaultContext
  }

  if (message.includes('预测') || message.includes('forecast') || message.includes('未来趋势')) {
    chatStore.sendPrediction(
      message,
      null,
      () => { pendingTool.value = null },
      (err) => ElMessage.error(err)
    )
  } else {
    chatStore.sendMessageWithContext(
      payload,
      null,
      () => { pendingTool.value = null },
      (err) => ElMessage.error(err)
    )
  }
}

function handleClear() {
  chatStore.clearHistory()
  ElMessage.success('对话已清空')
}

function handleExport() {
  chatStore.downloadMarkdown()
  ElMessage.success('导出成功')
}

function handleContextChange({ contexts, isManual }) {
  currentContexts.value = contexts
  isManualOverride.value = isManual
}

async function loadRecommendedQuestions() {
  try {
    const sessionId = sessionStore.currentSessionId
    if (!sessionId) {
      personalizedQuestions.value = {}
      return
    }

    const response = await api.get(`/session/${sessionId}/recommended_questions`)
    const data = response.data || response

    if (data && data.questions) {
      personalizedQuestions.value = data.questions
    } else if (data && typeof data === 'object') {
      personalizedQuestions.value = data
    } else {
      personalizedQuestions.value = {}
    }
  } catch (err) {
    console.warn('加载个性化推荐失败:', err)
    personalizedQuestions.value = {}
  }
}

async function refreshContext() {
  loadingContext.value = true
  try {
    const sessionId = sessionStore.currentSessionId
    if (sessionId) {
      const result = await reportApi.get(sessionId)
      reportData.value = result
    }
  } catch (err) {
    console.warn('刷新上下文失败:', err)
  } finally {
    loadingContext.value = false
  }
}

async function init() {
  const sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  chatStore.init(sessionId)
  await refreshContext()
  await loadRecommendedQuestions()
}

watch(() => sessionStore.currentSessionId, (newId) => {
  if (newId) {
    chatStore.init(newId)
    refreshContext()
    loadRecommendedQuestions()
  }
})

watch(() => route.path, () => {
  loadRecommendedQuestions()
})

onMounted(() => {
  init()
})
</script>

<style scoped>
.ai-assistant {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 16px;
}

.workbench-body {
  display: flex;
  flex: 1;
  min-height: 0;
  gap: 0;
  margin-top: 0;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.tool-panel-wrapper {
  width: 260px;
  flex-shrink: 0;
  height: 100%;
  overflow: hidden;
}

.chat-wrapper {
  flex: 1;
  height: 100%;
  overflow: hidden;
}

.context-panel-wrapper {
  width: 280px;
  flex-shrink: 0;
  height: 100%;
  overflow: hidden;
}

.footer-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-top: 1px solid #e4e7ed;
  flex-shrink: 0;
  margin-top: 8px;
}

@media (max-width: 1200px) {
  .context-panel-wrapper {
    width: 220px;
  }
}

@media (max-width: 992px) {
  .tool-panel-wrapper {
    width: 200px;
  }
  .context-panel-wrapper {
    width: 180px;
  }
}

@media (max-width: 768px) {
  .workbench-body {
    flex-direction: column;
  }
  .tool-panel-wrapper,
  .context-panel-wrapper {
    width: 100%;
    height: auto;
    max-height: 200px;
    overflow-y: auto;
  }
  .chat-wrapper {
    height: 400px;
    min-height: 300px;
  }
  .ai-assistant {
    padding: 0 8px;
  }
}
</style>