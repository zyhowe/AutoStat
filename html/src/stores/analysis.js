import { defineStore } from 'pinia'
import { ref } from 'vue'
import { analysisApi } from '../api/analysis'
import { reportApi } from '../api/report'
import { useSessionStore } from './session'

export const useAnalysisStore = defineStore('analysis', () => {
  const isLoading = ref(false)
  const progress = ref(0)
  const statusMessage = ref('')
  const taskId = ref(null)
  const qualityResult = ref(null)
  const reportData = ref(null)
  const summary = ref(null)
  const insights = ref(null)
  const error = ref(null)
  const analysisSessionId = ref(null)  // 🆕 单独存储分析用的 session_id

  async function runAnalysis(sessionId, variableTypes = {}) {
    isLoading.value = true
    progress.value = 0
    statusMessage.value = '提交分析任务...'
    error.value = null
    analysisSessionId.value = sessionId  // 🆕 保存

    try {
      const result = await analysisApi.run({
        session_id: sessionId,
        variable_types: variableTypes
      })

      taskId.value = result.task_id

      await pollStatus()
      await loadResults(sessionId)

      // 🆕 确保 session store 中的 session_id 正确
      const sessionStore = useSessionStore()
      if (sessionStore.currentSessionId !== sessionId) {
        console.log('修复 session_id:', sessionId)
        sessionStore.currentSessionId = sessionId
        // 重新加载会话信息
        await sessionStore.loadSession(sessionId)
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
    } catch (err) {
      console.error('加载分析结果失败:', err)
      throw err
    }
  }

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
  }

  return {
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
    runAnalysis,
    loadResults,
    reset
  }
})