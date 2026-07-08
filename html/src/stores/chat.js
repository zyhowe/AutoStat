import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { chatApi } from '../api/chat'

export const useChatStore = defineStore('chat', () => {
  // ============ State ============
  const messages = ref([])
  const isStreaming = ref(false)
  const streamingContent = ref('')
  const currentScene = ref('general')
  const contextData = ref({})
  const sessionId = ref(null)
  const isLoadingHistory = ref(false)
  const error = ref(null)

  // ============ Getters ============
  const hasMessages = computed(() => messages.value.length > 0)
  const lastMessage = computed(() => messages.value.length > 0 ? messages.value[messages.value.length - 1] : null)
  const messageCount = computed(() => messages.value.length)

  // ============ Actions ============

  function init(sid) {
    if (!sid) return
    sessionId.value = sid
    loadHistory(sid)
  }

  function loadHistory(sid) {
    const key = sid || sessionId.value
    if (!key) return

    isLoadingHistory.value = true
    error.value = null
    try {
      const stored = localStorage.getItem(`chat_history_${key}`)
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed)) {
          messages.value = parsed.slice(-100)
        } else {
          messages.value = []
        }
      } else {
        messages.value = []
      }
    } catch (e) {
      console.warn('加载对话历史失败:', e)
      messages.value = []
      error.value = '加载历史记录失败'
    } finally {
      isLoadingHistory.value = false
    }
  }

  function saveHistory() {
    const key = sessionId.value
    if (!key) return
    try {
      const data = messages.value.slice(-100)
      localStorage.setItem(`chat_history_${key}`, JSON.stringify(data))
    } catch (e) {
      console.warn('保存对话历史失败:', e)
    }
  }

  function clearHistory() {
    messages.value = []
    streamingContent.value = ''
    isStreaming.value = false
    error.value = null
    saveHistory()
  }

  function addUserMessage(text) {
    const msg = {
      role: 'user',
      content: text,
      time: new Date().toLocaleTimeString(),
      id: Date.now() + '_' + Math.random().toString(36).substr(2, 6)
    }
    messages.value.push(msg)
    saveHistory()
    return msg
  }

  function addAssistantMessage(content, extra = {}) {
    const msg = {
      role: 'assistant',
      content: content,
      time: new Date().toLocaleTimeString(),
      id: Date.now() + '_' + Math.random().toString(36).substr(2, 6),
      ...extra
    }
    messages.value.push(msg)
    saveHistory()
    return msg
  }

  // ===== 原 sendMessage（保持兼容） =====
  async function sendMessage(text, onChunk, onComplete, onError) {
    if (!text || !text.trim()) {
      if (onError) onError('请输入问题')
      return
    }
    if (!sessionId.value) {
      if (onError) onError('请先加载项目')
      return
    }

    addUserMessage(text.trim())

    isStreaming.value = true
    streamingContent.value = ''
    error.value = null

    try {
      await chatApi.chatStream(
        sessionId.value,
        text.trim(),
        ['json_result'],
        null,
        (chunk) => {
          streamingContent.value += chunk
          if (onChunk) onChunk(chunk)
        },
        () => {
          const fullContent = streamingContent.value
          addAssistantMessage(fullContent)
          streamingContent.value = ''
          isStreaming.value = false
          if (onComplete) onComplete(fullContent)
        },
        (err) => {
          isStreaming.value = false
          streamingContent.value = ''
          error.value = err
          if (onError) onError(err)
        }
      )
    } catch (err) {
      isStreaming.value = false
      streamingContent.value = ''
      error.value = err.message || '发送失败'
      if (onError) onError(error.value)
    }
  }

  // ===== 带上下文的 sendMessage =====
  async function sendMessageWithContext(payload, onChunk, onComplete, onError) {
    const { question, context_data } = payload

    if (!question || !question.trim()) {
      if (onError) onError('请输入问题')
      return
    }
    if (!sessionId.value) {
      if (onError) onError('请先加载项目')
      return
    }

    addUserMessage(question.trim())

    isStreaming.value = true
    streamingContent.value = ''
    error.value = null

    try {
      await chatApi.chatStream(
        sessionId.value,
        question.trim(),
        ['json_result'],
        context_data || null,
        (chunk) => {
          streamingContent.value += chunk
          if (onChunk) onChunk(chunk)
        },
        () => {
          const fullContent = streamingContent.value
          addAssistantMessage(fullContent)
          streamingContent.value = ''
          isStreaming.value = false
          if (onComplete) onComplete(fullContent)
        },
        (err) => {
          isStreaming.value = false
          streamingContent.value = ''
          error.value = err
          if (onError) onError(err)
        }
      )
    } catch (err) {
      isStreaming.value = false
      streamingContent.value = ''
      error.value = err.message || '发送失败'
      if (onError) onError(error.value)
    }
  }

  // ===== 预测 =====
  async function sendPrediction(text, onChunk, onComplete, onError) {
    if (!text || !text.trim()) {
      if (onError) onError('请输入预测需求')
      return
    }
    if (!sessionId.value) {
      if (onError) onError('请先加载项目')
      return
    }

    addUserMessage(text.trim())

    isStreaming.value = true
    streamingContent.value = ''
    error.value = null

    try {
      await chatApi.predictionStream(
        sessionId.value,
        text.trim(),
        (chunk) => {
          streamingContent.value += chunk
          if (onChunk) onChunk(chunk)
        },
        (result) => {
          const fullContent = streamingContent.value
          addAssistantMessage(fullContent, {
            isPrediction: true,
            predictionData: result
          })
          streamingContent.value = ''
          isStreaming.value = false
          if (onComplete) onComplete(fullContent, result)
        },
        (err) => {
          isStreaming.value = false
          streamingContent.value = ''
          error.value = err
          if (onError) onError(err)
        }
      )
    } catch (err) {
      isStreaming.value = false
      streamingContent.value = ''
      error.value = err.message || '预测失败'
      if (onError) onError(error.value)
    }
  }

  function setScene(scene, data = {}) {
    currentScene.value = scene
    contextData.value = data
  }

  function exportMarkdown() {
    if (messages.value.length === 0) return '# 对话记录\n\n暂无对话内容'

    let md = `# AI 对话记录\n\n`
    md += `- 导出时间: ${new Date().toLocaleString()}\n`
    md += `- 会话: ${sessionId.value || '未知'}\n`
    md += `- 消息总数: ${messages.value.length}\n\n---\n\n`

    messages.value.forEach((msg, index) => {
      const role = msg.role === 'user' ? '👤 **用户**' : '🤖 **AI 助手**'
      md += `### ${role}\n\n`
      md += `${msg.content}\n\n`
      if (msg.isPrediction) {
        md += `**预测结果**\n`
        if (msg.predictionData?.prediction !== undefined) {
          md += `- 预测值: ${msg.predictionData.prediction}\n`
        }
        if (msg.predictionData?.confidence !== undefined) {
          md += `- 置信度: ${(msg.predictionData.confidence * 100).toFixed(1)}%\n`
        }
        if (msg.predictionData?.model_used) {
          md += `- 模型: ${msg.predictionData.model_used}\n`
        }
        md += '\n'
      }
      if (msg.time) md += `*${msg.time}*\n\n`
      if (index < messages.value.length - 1) md += '---\n\n'
    })

    return md
  }

  function downloadMarkdown() {
    const content = exportMarkdown()
    if (!content || content === '# 对话记录\n\n暂无对话内容') {
      return
    }
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `对话记录_${new Date().toISOString().slice(0, 10)}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function reset() {
    messages.value = []
    isStreaming.value = false
    streamingContent.value = ''
    sessionId.value = null
    currentScene.value = 'general'
    contextData.value = {}
    error.value = null
  }

  return {
    messages,
    isStreaming,
    streamingContent,
    currentScene,
    contextData,
    sessionId,
    isLoadingHistory,
    error,
    hasMessages,
    lastMessage,
    messageCount,
    init,
    loadHistory,
    saveHistory,
    clearHistory,
    addUserMessage,
    addAssistantMessage,
    sendMessage,
    sendMessageWithContext,
    sendPrediction,
    setScene,
    exportMarkdown,
    downloadMarkdown,
    reset
  }
})