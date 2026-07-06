<template>
  <div class="global-bubble" ref="bubbleRef">
    <!-- 气泡按钮 -->
    <div
      class="bubble-btn"
      :class="{ active: panelVisible || fullscreenVisible, hasUnread: hasUnread }"
      @click.stop="togglePanel"
    >
      <el-icon v-if="!panelVisible && !fullscreenVisible"><ChatDotRound /></el-icon>
      <el-icon v-else><Close /></el-icon>
      <span class="bubble-tip">问AI</span>
      <span v-if="hasUnread" class="unread-dot"></span>
      <span class="pulse-ring"></span>
    </div>

    <!-- 气泡面板（小窗模式） -->
    <transition name="slide-up">
      <div v-if="panelVisible && !fullscreenVisible" class="bubble-panel" @click.stop>
        <div class="panel-header">
          <span class="panel-title">🤖 AI 助手</span>
          <div class="header-actions">
            <el-button size="small" text @click="openFullscreen">⛶ 放大</el-button>
            <el-button size="small" text @click="clearChat">清空</el-button>
            <el-button size="small" text @click="panelVisible = false">✕</el-button>
          </div>
        </div>

        <!-- 上下文选择器 -->
        <div class="context-selector">
          <span class="context-label">📌 上下文：</span>
          <el-select
            v-model="selectedContext"
            size="small"
            class="context-select"
            @change="onContextChange"
          >
            <el-option
              v-for="(label, key) in contextOptions"
              :key="key"
              :label="label"
              :value="key"
            />
          </el-select>
          <el-tag v-if="isPredictionMode" size="small" type="warning" style="margin-left: 8px;">
            🤖 预测模式
          </el-tag>
        </div>

        <!-- 对话区域 -->
        <div class="chat-messages" ref="chatContainer">
          <div v-if="messages.length === 0" class="chat-empty">
            <p>👋 你好！我是 AI 助手</p>
            <p class="hint">选择上下文后，从下方选择推荐问题开始</p>
          </div>
          <div
            v-for="(msg, index) in messages"
            :key="index"
            class="chat-msg"
            :class="msg.role"
          >
            <span class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</span>
            <div class="msg-content-wrapper">
              <div class="msg-content" v-html="formatMessage(msg.content)"></div>
              <div v-if="msg.role === 'assistant'" class="msg-time">{{ msg.time }}</div>
            </div>
          </div>
          <div v-if="isStreaming" class="chat-msg assistant">
            <span class="msg-avatar">🤖</span>
            <div class="msg-content-wrapper">
              <div class="msg-content streaming">{{ streamingContent }}<span class="cursor">▌</span></div>
            </div>
          </div>
        </div>

        <!-- 推荐问题 -->
        <div class="quick-questions">
          <div class="quick-header">
            <span>💡 推荐问题</span>
            <el-button size="small" text @click="showAllQuestions = !showAllQuestions">
              {{ showAllQuestions ? '收起' : '展开全部' }}
            </el-button>
          </div>
          <div v-if="loadingQuestions" class="quick-loading">
            <el-skeleton :rows="2" animated />
          </div>
          <div v-else class="quick-list" :class="{ expanded: showAllQuestions }">
            <span
              v-for="(q, index) in displayQuestions"
              :key="index"
              class="quick-item"
              @click="handleQuestionClick(q)"
            >
              {{ q.icon }} {{ q.text }}
            </span>
            <div v-if="displayQuestions.length === 0" class="quick-empty">
              暂无推荐问题，请先完成数据分析
            </div>
          </div>
        </div>

        <!-- 输入框 -->
        <div class="chat-input">
          <el-input
            v-model="inputQuestion"
            size="default"
            :placeholder="inputPlaceholder"
            @keyup.enter="handleSend"
          >
            <template #append>
              <el-button type="primary" :loading="isStreaming" @click="handleSend">
                {{ isPredictionMode ? '预测' : '发送' }}
              </el-button>
            </template>
          </el-input>
        </div>
      </div>
    </transition>

    <!-- 全屏对话框 -->
    <transition name="fade">
      <div v-if="fullscreenVisible" class="bubble-fullscreen" @click.stop>
        <div class="fullscreen-header">
          <span class="fullscreen-title">🤖 AI 助手</span>
          <div class="fullscreen-actions">
            <el-button size="small" text @click="closeFullscreen">⛶ 缩小</el-button>
            <el-button size="small" text @click="clearChat">清空</el-button>
            <el-button size="small" text @click="fullscreenVisible = false">✕</el-button>
          </div>
        </div>

        <div class="fullscreen-body">
          <div class="context-selector full">
            <span class="context-label">📌 上下文：</span>
            <el-select
              v-model="selectedContext"
              size="default"
              class="context-select full"
              @change="onContextChange"
            >
              <el-option
                v-for="(label, key) in contextOptions"
                :key="key"
                :label="label"
                :value="key"
              />
            </el-select>
            <el-tag v-if="isPredictionMode" size="small" type="warning" style="margin-left: 8px;">
              🤖 预测模式
            </el-tag>
          </div>

          <div class="chat-messages full" ref="fullChatContainer">
            <div v-if="messages.length === 0" class="chat-empty">
              <p>👋 你好！我是 AI 助手</p>
              <p class="hint">选择上下文后，从下方选择推荐问题开始</p>
            </div>
            <div
              v-for="(msg, index) in messages"
              :key="index"
              class="chat-msg"
              :class="msg.role"
            >
              <span class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</span>
              <div class="msg-content-wrapper">
                <div class="msg-content" v-html="formatMessage(msg.content)"></div>
                <div v-if="msg.role === 'assistant'" class="msg-time">{{ msg.time }}</div>
              </div>
            </div>
            <div v-if="isStreaming" class="chat-msg assistant">
              <span class="msg-avatar">🤖</span>
              <div class="msg-content-wrapper">
                <div class="msg-content streaming">{{ streamingContent }}<span class="cursor">▌</span></div>
              </div>
            </div>
          </div>

          <div class="quick-questions full">
            <div class="quick-header">
              <span>💡 推荐问题</span>
            </div>
            <div v-if="loadingQuestions" class="quick-loading">
              <el-skeleton :rows="2" animated />
            </div>
            <div v-else class="quick-list full">
              <span
                v-for="(q, index) in allPageQuestions"
                :key="index"
                class="quick-item"
                @click="handleQuestionClick(q)"
              >
                {{ q.icon }} {{ q.text }}
              </span>
              <div v-if="allPageQuestions.length === 0" class="quick-empty">
                暂无推荐问题
              </div>
            </div>
          </div>

          <div class="chat-input full">
            <el-input
              v-model="inputQuestion"
              size="large"
              :placeholder="inputPlaceholder"
              @keyup.enter="handleSend"
            >
              <template #append>
                <el-button type="primary" :loading="isStreaming" @click="handleSend">
                  {{ isPredictionMode ? '预测' : '发送' }}
                </el-button>
              </template>
            </el-input>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ChatDotRound, Close } from '@element-plus/icons-vue'
import { useSessionStore } from '../stores/session'
import { chatApi } from '../api/chat'
import axios from 'axios'

const route = useRoute()
const sessionStore = useSessionStore()
const bubbleRef = ref(null)
const chatContainer = ref(null)
const fullChatContainer = ref(null)

// ==================== 状态 ====================
const panelVisible = ref(false)
const fullscreenVisible = ref(false)
const inputQuestion = ref('')
const isStreaming = ref(false)
const streamingContent = ref('')
const messages = ref([])
const showAllQuestions = ref(false)
const hasUnread = ref(false)
const selectedContext = ref('')
const loadingQuestions = ref(false)
const backendQuestions = ref({})

// ==================== 上下文选项 ====================
const contextOptions = {
  '/upload': '上传数据',
  '/report-summary': '报告总览',
  '/data-overview': '数据概览',
  '/quality': '质量看板',
  '/data-validation': '数据核验',
  '/pattern-discovery': '规律发现',
  '/models': '智能预测',
  '/settings': '设置'
}

// 场景映射（后端存储的key -> 前端场景key）
const sceneMap = {
  '/upload': 'upload',
  '/report-summary': 'report_summary',
  '/data-overview': 'data_overview',
  '/quality': 'quality',
  '/data-validation': 'data_validation',
  '/pattern-discovery': 'pattern_discovery',
  '/models': 'smart_prediction',
  '/settings': 'settings'
}

// ==================== 计算属性 ====================

const isPredictionMode = computed(() => {
  return route.path === '/models'
})

const inputPlaceholder = computed(() => {
  if (isPredictionMode.value) {
    return '输入预测需求，如：预测下个月的销售额'
  }
  return '输入问题...'
})

const currentScene = computed(() => {
  return sceneMap[route.path] || 'report_summary'
})

// 当前选中的上下文对应的推荐问题（从后端加载）
const currentQuestions = computed(() => {
  const scene = currentScene.value
  return backendQuestions.value[scene] || []
})

const allPageQuestions = computed(() => {
  return currentQuestions.value
})

const displayQuestions = computed(() => {
  const qs = currentQuestions.value
  return showAllQuestions.value ? qs : qs.slice(0, 4)
})

// ==================== 加载推荐问题 ====================

async function loadRecommendedQuestions() {
  const sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    return
  }

  loadingQuestions.value = true
  try {
    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'}/chat/recommended_questions`,
      { params: { session_id: sessionId } }
    )
    backendQuestions.value = response.data || {}
  } catch (err) {
    console.warn('加载推荐问题失败:', err)
    // 降级到默认问题
    backendQuestions.value = getDefaultQuestions()
  } finally {
    loadingQuestions.value = false
  }
}

function getDefaultQuestions() {
  return {
    report_summary: [
      { icon: '📊', text: '总结报告核心结论' },
      { icon: '🔍', text: '数据质量整体评价' }
    ],
    data_overview: [
      { icon: '📋', text: '解释字段类型分布' },
      { icon: '🔍', text: '数据完整性如何？' }
    ],
    quality: [
      { icon: '📊', text: '解读质量评分' },
      { icon: '🔍', text: '各维度得分偏低的原因' }
    ],
    data_validation: [
      { icon: '🔗', text: '解释勾稽规则的含义' },
      { icon: '🚨', text: '异常值可能的原因分析' }
    ],
    pattern_discovery: [
      { icon: '🔗', text: '解读相关性分析结果' },
      { icon: '📈', text: '时间序列预测建议' }
    ],
    smart_prediction: [
      { icon: '🔮', text: '预测下个月的销售额' },
      { icon: '🤖', text: '推荐适合的预测模型' },
      { icon: '📊', text: '对主要指标做未来趋势预测' }
    ]
  }
}

// ==================== 初始化上下文 ====================

function initContext() {
  const path = route.path
  if (contextOptions[path]) {
    selectedContext.value = path
  } else {
    selectedContext.value = Object.keys(contextOptions)[0] || '/upload'
  }
}

function onContextChange() {
  showAllQuestions.value = false
  // 切换上下文时重新加载推荐问题
  loadRecommendedQuestions()
  ElMessage.info(`已切换到「${contextOptions[selectedContext.value]}」上下文`)
}

// ==================== 对话持久化 ====================

const STORAGE_KEY = 'global_bubble_chat_history'

function loadHistory() {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    if (data) {
      const parsed = JSON.parse(data)
      // 只保留最近50条
      messages.value = parsed.slice(-50)
    }
  } catch (e) {
    console.warn('加载对话历史失败:', e)
  }
}

function saveHistory() {
  try {
    // 只保留最近50条
    const data = messages.value.slice(-50)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch (e) {
    console.warn('保存对话历史失败:', e)
  }
}

// ==================== 核心方法 ====================

function togglePanel() {
  panelVisible.value = !panelVisible.value
  if (panelVisible.value) {
    hasUnread.value = false
    loadHistory()
    loadRecommendedQuestions()
    scrollToBottom(chatContainer)
  }
}

function openFullscreen() {
  panelVisible.value = false
  fullscreenVisible.value = true
  nextTick(() => {
    scrollToBottom(fullChatContainer)
  })
}

function closeFullscreen() {
  fullscreenVisible.value = false
  panelVisible.value = true
  nextTick(() => {
    scrollToBottom(chatContainer)
  })
}

function handleQuestionClick(q) {
  const text = q.text
  inputQuestion.value = text
  handleSend()
}

async function handleSend() {
  const question = inputQuestion.value.trim()
  if (!question) return

  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  const timestamp = new Date().toLocaleTimeString()
  messages.value.push({ role: 'user', content: question, time: timestamp })
  inputQuestion.value = ''
  saveHistory()
  scrollToBottom(chatContainer)
  scrollToBottom(fullChatContainer)

  isStreaming.value = true
  streamingContent.value = ''

  try {
    // 判断是否是预测模式（智能预测页面 + 预测类问题）
    if (isPredictionMode.value && isPredictionQuestion(question)) {
      await handlePrediction(question)
    } else {
      await handleNormalChat(question)
    }
  } catch (err) {
    ElMessage.error('对话失败: ' + err.message)
    isStreaming.value = false
  }
}

function isPredictionQuestion(question) {
  const keywords = ['预测', '预估', '预报', '推演', 'forecast', 'predict']
  const qLower = question.toLowerCase()
  return keywords.some(kw => qLower.includes(kw))
}

async function handlePrediction(question) {
  try {
    const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'
    const response = await axios.post(`${baseURL}/chat/prediction/stream`, {
      session_id: sessionStore.currentSessionId,
      question: question
    }, {
      responseType: 'stream'
    })

    const reader = response.data.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6))
            if (data.content) {
              streamingContent.value += data.content
              scrollToBottom(chatContainer)
              scrollToBottom(fullChatContainer)
            }
            if (data.done) {
              const ts = new Date().toLocaleTimeString()
              messages.value.push({
                role: 'assistant',
                content: streamingContent.value,
                time: ts,
                isPrediction: true,
                data: data.data,
                model_used: data.model_used,
                confidence: data.confidence
              })
              streamingContent.value = ''
              isStreaming.value = false
              saveHistory()
              scrollToBottom(chatContainer)
              scrollToBottom(fullChatContainer)
            }
            if (data.error) {
              ElMessage.error('预测失败: ' + data.error)
              isStreaming.value = false
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }
  } catch (err) {
    ElMessage.error('预测失败: ' + err.message)
    isStreaming.value = false
  }
}

async function handleNormalChat(question) {
  const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'

  const response = await fetch(`${baseURL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionStore.currentSessionId,
      question: question,
      context: ['json_result']
    })
  })

  if (!response.ok) {
    throw new Error('请求失败')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.substring(6))
          if (data.content) {
            streamingContent.value += data.content
            scrollToBottom(chatContainer)
            scrollToBottom(fullChatContainer)
          }
          if (data.done) {
            const ts = new Date().toLocaleTimeString()
            messages.value.push({ role: 'assistant', content: streamingContent.value, time: ts })
            streamingContent.value = ''
            isStreaming.value = false
            saveHistory()
            scrollToBottom(chatContainer)
            scrollToBottom(fullChatContainer)
          }
        } catch (e) {
          // 忽略解析错误
        }
      }
    }
  }
}

function clearChat() {
  messages.value = []
  streamingContent.value = ''
  isStreaming.value = false
  saveHistory()
  ElMessage.success('对话已清空')
}

function scrollToBottom(ref) {
  nextTick(() => {
    if (ref?.value) {
      ref.value.scrollTop = ref.value.scrollHeight
    }
  })
}

function handleClickOutside(event) {
  if (bubbleRef.value && !bubbleRef.value.contains(event.target)) {
    panelVisible.value = false
    fullscreenVisible.value = false
  }
}

function formatMessage(content) {
  if (!content) return ''
  return content
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
}

// ==================== 生命周期 ====================

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  loadHistory()
  initContext()
  if (sessionStore.hasSession) {
    loadRecommendedQuestions()
  }
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
})

watch(() => route.path, (newPath) => {
  if (contextOptions[newPath]) {
    selectedContext.value = newPath
    showAllQuestions.value = false
    if (sessionStore.hasSession) {
      loadRecommendedQuestions()
    }
  }
})

watch(() => sessionStore.currentSessionId, (newId) => {
  if (newId) {
    loadRecommendedQuestions()
  }
})
</script>

<style scoped>
.global-bubble {
  position: fixed;
  bottom: 40px;
  right: 40px;
  z-index: 9999;
}

/* ==================== 气泡按钮 ==================== */
.bubble-btn {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 24px rgba(102, 126, 234, 0.45);
  transition: all 0.3s ease;
  position: relative;
  user-select: none;
}

.bubble-btn:hover {
  transform: scale(1.08);
  box-shadow: 0 6px 36px rgba(102, 126, 234, 0.55);
}

.bubble-btn .el-icon {
  font-size: 30px;
  z-index: 2;
}

.bubble-btn.active {
  background: #f56c6c;
}

.bubble-btn.active:hover {
  transform: scale(1.05);
}

.pulse-ring {
  position: absolute;
  top: -6px;
  left: -6px;
  right: -6px;
  bottom: -6px;
  border-radius: 50%;
  border: 3px solid rgba(102, 126, 234, 0.3);
  animation: pulse 2s ease-out infinite;
  pointer-events: none;
}

@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 0.8;
  }
  100% {
    transform: scale(1.25);
    opacity: 0;
  }
}

.bubble-tip {
  position: absolute;
  top: -10px;
  right: -10px;
  background: #ff6b6b;
  color: white;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 12px;
  font-weight: bold;
  z-index: 3;
}

.unread-dot {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 12px;
  height: 12px;
  background: #ff6b6b;
  border-radius: 50%;
  border: 2px solid white;
  z-index: 3;
  animation: blink-dot 1s ease-in-out infinite;
}

@keyframes blink-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* ==================== 气泡面板 ==================== */
.bubble-panel {
  position: absolute;
  bottom: 80px;
  right: 0;
  width: 440px;
  max-height: 640px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 48px rgba(0, 0, 0, 0.18);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  background: #fafafa;
  flex-shrink: 0;
}

.panel-title {
  font-size: 15px;
  font-weight: 600;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 4px;
}

.header-actions .el-button {
  font-size: 12px;
  padding: 4px 8px;
}

/* 上下文选择器 */
.context-selector {
  display: flex;
  align-items: center;
  padding: 6px 16px;
  background: #f5f7fa;
  border-bottom: 1px solid #e8ecf1;
  flex-shrink: 0;
  gap: 8px;
}

.context-selector.full {
  padding: 10px 20px;
  background: #f5f7fa;
}

.context-label {
  font-size: 12px;
  color: #666;
  white-space: nowrap;
}

.context-select {
  width: 150px;
}

.context-select.full {
  width: 200px;
}

.context-select :deep(.el-input__wrapper) {
  background: white;
  border-radius: 6px;
}

.context-select :deep(.el-input__inner) {
  font-size: 12px;
}

/* 对话区域 */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
  min-height: 120px;
  max-height: 240px;
}

.chat-messages.full {
  max-height: none;
  min-height: 200px;
  flex: 1;
}

.chat-empty {
  text-align: center;
  color: #bbb;
  padding: 30px 0;
}

.chat-empty p {
  margin: 4px 0;
  font-size: 14px;
}

.chat-empty .hint {
  font-size: 12px;
  color: #ccc;
}

.chat-msg {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
  align-items: flex-start;
}

.chat-msg.user {
  flex-direction: row-reverse;
}

.chat-msg.user .msg-content {
  background: #409eff;
  color: white;
  padding: 8px 14px;
  border-radius: 14px 14px 4px 14px;
}

.chat-msg.assistant .msg-content {
  background: #f0f2f6;
  padding: 8px 14px;
  border-radius: 14px 14px 14px 4px;
}

.msg-avatar {
  font-size: 18px;
  flex-shrink: 0;
  width: 28px;
  text-align: center;
}

.msg-content-wrapper {
  max-width: 80%;
}

.msg-content {
  font-size: 13px;
  line-height: 1.6;
  word-break: break-word;
}

.msg-content.streaming {
  white-space: pre-wrap;
}

.msg-time {
  font-size: 10px;
  color: #bbb;
  margin-top: 2px;
  text-align: right;
}

.cursor {
  animation: blink 0.8s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* 推荐问题 */
.quick-questions {
  border-top: 1px solid #f0f0f0;
  padding: 8px 12px 10px 12px;
  background: #fafafa;
  flex-shrink: 0;
  max-height: 200px;
  display: flex;
  flex-direction: column;
}

.quick-questions.full {
  max-height: none;
  flex-shrink: 0;
}

.quick-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.quick-header span {
  font-size: 12px;
  font-weight: 500;
  color: #666;
}

.quick-header .el-button {
  font-size: 11px;
  padding: 2px 6px;
}

.quick-loading {
  padding: 8px 0;
}

.quick-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  max-height: 56px;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.quick-list.expanded {
  max-height: 300px;
  overflow-y: auto;
}

.quick-list.full {
  max-height: none;
  overflow-y: auto;
}

.quick-item {
  font-size: 12px;
  color: #409eff;
  background: white;
  padding: 3px 12px;
  border-radius: 16px;
  border: 1px solid #d9ecff;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.quick-item:hover {
  background: #ecf5ff;
  border-color: #409eff;
}

.quick-empty {
  font-size: 12px;
  color: #bbb;
  padding: 8px 0;
  width: 100%;
  text-align: center;
}

/* 输入框 */
.chat-input {
  padding: 8px 12px 12px 12px;
  border-top: 1px solid #f0f0f0;
  background: white;
  flex-shrink: 0;
}

.chat-input.full {
  padding: 12px 20px 20px 20px;
}

.chat-input .el-input {
  width: 100%;
}

/* ==================== 全屏模式 ==================== */
.bubble-fullscreen {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: white;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  animation: fadeIn 0.25s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.fullscreen-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid #e4e7ed;
  background: #fafafa;
  flex-shrink: 0;
}

.fullscreen-title {
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.fullscreen-actions {
  display: flex;
  gap: 8px;
}

.fullscreen-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 0 24px 24px 24px;
  overflow: hidden;
}

/* 动画 */
.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.3s ease;
}

.slide-up-enter-from,
.slide-up-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 响应式 */
@media (max-width: 768px) {
  .bubble-btn {
    width: 54px;
    height: 54px;
  }
  .bubble-btn .el-icon {
    font-size: 24px;
  }
  .bubble-panel {
    width: 330px;
    right: -10px;
    max-height: 540px;
  }
  .bubble-panel .chat-messages {
    max-height: 160px;
  }
  .context-select {
    width: 100px;
  }
  .fullscreen-header {
    padding: 12px 16px;
  }
  .fullscreen-title {
    font-size: 16px;
  }
  .fullscreen-body {
    padding: 0 12px 12px 12px;
  }
  .context-select.full {
    width: 140px;
  }
}
</style>