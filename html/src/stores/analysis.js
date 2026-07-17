import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { analysisApi } from '../api/analysis'
import { reportApi } from '../api/report'
import { useSessionStore } from './session'

export const useAnalysisStore = defineStore('analysis', () => {
  // ==================== 原有状态 ====================
  const isLoading = ref(false)
  const progress = ref(0)
  const statusMessage = ref('')
  const taskId = ref(null)
  const qualityResult = ref(null)
  const reportData = ref(null)
  const summary = ref(null)
  const insights = ref(null)
  const error = ref(null)
  const analysisSessionId = ref(null)
  let router = null

  // ==================== 新增：表选择器状态 ====================
  const currentTable = ref('merged')
  const tableNames = ref([])
  const isMultiTable = ref(false)

  // ==================== 原有 Getters ====================
  const hasSession = computed(() => !!analysisSessionId.value)

  // ==================== 新增：表选择器 Getters ====================
  const currentData = computed(() => {
    if (!reportData.value?.all_tables) return {}
    return reportData.value.all_tables[currentTable.value] || reportData.value.all_tables['merged'] || {}
  })

  const tableOptions = computed(() => {
    const options = []
    options.push({ label: '📊 合并表', value: 'merged' })
    for (const name of tableNames.value) {
      options.push({ label: `📋 ${name}`, value: name })
    }
    return options
  })

  // ==================== 原有 Actions ====================
  function setRouter(routerInstance) {
    router = routerInstance
  }

  // ===== 修改：runAnalysis 不再接收 variable_types =====
  async function runAnalysis(sessionId) {
    isLoading.value = true
    progress.value = 0
    statusMessage.value = '提交分析任务...'
    error.value = null
    analysisSessionId.value = sessionId

    try {
      // 不再传递 variable_types，后端从缓存读取
      const result = await analysisApi.run({
        session_id: sessionId,
        include_html: false
      })

      taskId.value = result.task_id

      await pollStatus()
      await loadResults(sessionId)

      const sessionStore = useSessionStore()
      if (sessionStore.currentSessionId !== sessionId) {
        console.log('修复 session_id:', sessionId)
        sessionStore.currentSessionId = sessionId
        await sessionStore.loadSession(sessionId)
      }

      await sessionStore.loadProjects()

      if (router) {
        router.push('/report-summary')
      }

      return true
    } catch (err) {
      error.value = err.message
      return false
    } finally {
      isLoading.value = false
    }
  }

  async function pollStatus() {
    if (!taskId.value) return

    const maxAttempts = 120
    let attempts = 0

    while (attempts < maxAttempts) {
      const status = await analysisApi.getStatus(taskId.value)

      progress.value = status.progress || 0
      statusMessage.value = status.message || '处理中...'

      if (status.status === 'completed') {
        return true
      }

      if (status.status === 'failed') {
        throw new Error(status.error || '分析失败')
      }

      attempts++
      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    throw new Error('分析超时')
  }

  async function loadResults(sessionId) {
    try {
      const [quality, report, summaryData, insightsData] = await Promise.all([
        reportApi.getQuality(sessionId),
        reportApi.get(sessionId),
        reportApi.getSummary(sessionId),
        reportApi.getInsights(sessionId)
      ])

      qualityResult.value = quality
      reportData.value = report
      summary.value = summaryData.conclusions || []
      insights.value = insightsData

      // ✅ 初始化表选择器
      initTableSelector(report)
    } catch (err) {
      console.error('加载分析结果失败:', err)
      throw err
    }
  }

  // ==================== 新增：表选择器 Actions ====================
  function initTableSelector(analysisResult) {
    const allTables = analysisResult?.all_tables || {}
    const keys = Object.keys(allTables)
    tableNames.value = keys.filter(k => k !== 'merged')
    isMultiTable.value = tableNames.value.length > 1

    // 如果当前表不存在于 all_tables 中，重置为 merged
    if (!currentTable.value || !allTables[currentTable.value]) {
      currentTable.value = 'merged'
    }
  }

  function setCurrentTable(table) {
    if (table && reportData.value?.all_tables?.[table]) {
      currentTable.value = table
    }
  }

  // ==================== 原有 Actions ====================
  function reset() {
    isLoading.value = false
    progress.value = 0
    statusMessage.value = ''
    taskId.value = null
    qualityResult.value = null
    reportData.value = null
    summary.value = null
    insights.value = null
    error.value = null
    analysisSessionId.value = null

    // 重置表选择器状态
    currentTable.value = 'merged'
    tableNames.value = []
    isMultiTable.value = false
  }

  // ==================== 导出 ====================
  return {
    // 原有状态
    isLoading,
    progress,
    statusMessage,
    taskId,
    qualityResult,
    reportData,
    summary,
    insights,
    error,
    analysisSessionId,

    // 新增状态
    currentTable,
    tableNames,
    isMultiTable,

    // 原有 Getters
    hasSession,

    // 新增 Getters
    currentData,
    tableOptions,

    // 原有 Actions
    setRouter,
    runAnalysis,
    pollStatus,
    loadResults,
    reset,

    // 新增 Actions
    initTableSelector,
    setCurrentTable
  }
})