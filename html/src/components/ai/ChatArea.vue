<template>
  <div class="chat-area">
    <!-- ===== 对话头部 ===== -->
    <div class="chat-header">
      <div class="header-left">
        <span class="chat-title">💬 对话</span>
        <el-radio-group v-model="selectedContext" @change="onContextChange" size="small" class="context-radio">
          <el-radio-button value="upload">📁 上传数据</el-radio-button>
          <el-radio-button value="source">🗄️ 源数据</el-radio-button>
        </el-radio-group>
      </div>
      <div class="header-actions">
        <el-button size="small" text @click="scrollToBottom">⬇ 到底</el-button>
        <el-button size="small" text @click="emit('clear')">清空</el-button>
      </div>
    </div>

    <!-- ===== 消息列表 ===== -->
    <div class="chat-messages" ref="messagesRef">
      <div v-if="messages.length === 0 && !isStreaming" class="empty-state">
        <div class="empty-icon">🤖</div>
        <p class="empty-title">开始你的分析</p>
        <p class="empty-desc">从右侧选择推荐问题或工具，或者直接在下方输入</p>
      </div>

      <div
        v-for="msg in messages"
        :key="msg.id"
        class="message-wrapper"
        :class="msg.role"
      >
        <div class="message-avatar">
          {{ msg.role === 'user' ? '👤' : '🤖' }}
        </div>
        <div class="message-bubble">
          <div class="message-content" v-html="formatContent(msg.content)"></div>
          <div class="message-footer">
            <span class="message-time">{{ msg.time }}</span>
            <div v-if="msg.isPrediction && msg.predictionData" class="prediction-card">
              <div class="prediction-row">
                <span class="prediction-label">📊 预测值</span>
                <span class="prediction-value">{{ msg.predictionData.prediction }}</span>
              </div>
              <div v-if="msg.predictionData.confidence !== undefined && msg.predictionData.confidence !== null" class="prediction-row">
                <span class="prediction-label">📈 置信度</span>
                <span class="prediction-value confidence">{{ (msg.predictionData.confidence * 100).toFixed(1) }}%</span>
              </div>
              <div v-if="msg.predictionData.model_used" class="prediction-row">
                <span class="prediction-label">🤖 模型</span>
                <span class="prediction-value model">{{ msg.predictionData.model_used }}</span>
              </div>
              <div v-if="msg.predictionData.probabilities && msg.predictionData.probabilities.length > 0" class="prediction-probs">
                <span class="prediction-label">📊 概率分布</span>
                <div class="prob-bars">
                  <div
                    v-for="(p, idx) in msg.predictionData.probabilities"
                    :key="idx"
                    class="prob-bar-item"
                  >
                    <span class="prob-label">类别 {{ idx }}</span>
                    <el-progress
                      :percentage="p * 100"
                      :stroke-width="6"
                      :show-text="false"
                      :color="p > 0.5 ? '#67c23a' : '#409eff'"
                    />
                    <span class="prob-value">{{ (p * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="isStreaming" class="message-wrapper assistant">
        <div class="message-avatar">🤖</div>
        <div class="message-bubble streaming-bubble">
          <div class="message-content" v-html="formatContent(streamingContent)"></div>
          <span class="cursor">▌</span>
        </div>
      </div>
    </div>

    <!-- ===== 输入框 ===== -->
    <div class="chat-input-area">
      <div class="input-wrapper">
        <el-input
          ref="inputRef"
          v-model="inputText"
          :placeholder="inputPlaceholder"
          size="large"
          @keyup.enter="handleSend"
          @focus="onInputFocus"
          clearable
          class="chat-input"
        >
          <template #append>
            <el-button
              type="primary"
              :loading="isStreaming"
              @click="handleSend"
              class="send-btn"
            >
              {{ isStreaming ? '生成中' : '发送' }}
            </el-button>
          </template>
        </el-input>
      </div>
      <div class="input-hint" v-if="pendingTool">
        <span class="tool-hint">
          🛠️ {{ pendingTool.name }}
          <el-button size="small" text @click="clearPendingTool">取消</el-button>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch, onMounted } from 'vue'

const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  isStreaming: {
    type: Boolean,
    default: false
  },
  streamingContent: {
    type: String,
    default: ''
  },
  pendingTool: {
    type: Object,
    default: null
  },
  contextValue: {
    type: String,
    default: 'upload'
  }
})

const emit = defineEmits(['send', 'clear', 'tool-clear', 'context-change'])

const inputText = ref('')
const inputRef = ref(null)
const messagesRef = ref(null)
const inputPlaceholder = ref('输入问题...')

const selectedContext = ref(props.contextValue)

watch(() => props.contextValue, (newVal) => {
  if (newVal) {
    selectedContext.value = newVal
  }
})

function onContextChange(value) {
  emit('context-change', value)
}

function handleSend() {
  const text = inputText.value.trim()
  if (!text) return
  emit('send', text)
  inputText.value = ''
  if (props.pendingTool) {
    emit('tool-clear')
  }
}

function onInputFocus() {}

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

function formatContent(content) {
  if (!content) return ''
  let formatted = content
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
  formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`
  })
  return formatted
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

function clearPendingTool() {
  emit('tool-clear')
  inputPlaceholder.value = '输入问题...'
}

watch(() => props.pendingTool, (tool) => {
  if (tool) {
    if (tool.id === 'execute_predict') {
      inputText.value = tool.prompt || '请预测 '
    } else {
      inputText.value = tool.prompt || ''
    }
    inputPlaceholder.value = `🛠️ ${tool.name}`
    nextTick(() => {
      inputRef.value?.focus()
      const input = inputRef.value?.$el?.querySelector('input')
      if (input) {
        input.setSelectionRange(input.value.length, input.value.length)
      }
    })
  }
}, { immediate: true })

watch(() => props.messages.length, () => scrollToBottom(), { deep: true })
watch(() => props.streamingContent, () => scrollToBottom())

onMounted(() => scrollToBottom())

defineExpose({ scrollToBottom, focus: () => inputRef.value?.focus() })
</script>

<style scoped>
.chat-area {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #fff;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid #e4e7ed;
  flex-shrink: 0;
  flex-wrap: wrap;
  gap: 8px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.chat-title {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
}

.context-radio {
  flex-shrink: 0;
}

.context-radio :deep(.el-radio-button__inner) {
  font-size: 12px;
  padding: 4px 12px;
}

.header-actions {
  display: flex;
  gap: 4px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #bbb;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-title {
  font-size: 18px;
  font-weight: 500;
  color: #909399;
  margin-bottom: 8px;
}

.empty-desc {
  font-size: 14px;
  color: #c0c4cc;
}

.message-wrapper {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  align-items: flex-start;
  width: 100%;
}

.message-wrapper.user {
  flex-direction: row-reverse;
}

.message-wrapper.user .message-bubble {
  background: #409eff;
  color: white;
  border-radius: 14px 14px 4px 14px;
}

.message-wrapper.assistant {
  margin-right: auto;
}

.message-wrapper.assistant .message-bubble {
  background: #f0f2f6;
  border-radius: 14px 14px 14px 4px;
}

.message-avatar {
  font-size: 20px;
  flex-shrink: 0;
  width: 32px;
  text-align: center;
}

.message-bubble {
  max-width: 96%;
  padding: 10px 14px;
  word-break: break-word;
  line-height: 1.5;
  font-size: 13px;
}

.message-bubble.streaming-bubble {
  background: #f0f2f6;
  border-radius: 14px 14px 14px 4px;
}

.message-content {
  white-space: pre-wrap;
  font-size: 13px;
  line-height: 1.5;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.message-content :deep(pre) {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 11px;
  margin: 6px 0;
  font-family: 'Consolas', 'Courier New', monospace;
}

.message-content :deep(code) {
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 11px;
}

.message-footer {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.message-time {
  font-size: 10px;
  color: #bbb;
  text-align: right;
}

.cursor {
  animation: blink 0.8s infinite;
  color: #409eff;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.prediction-card {
  margin-top: 8px;
  padding: 10px 14px;
  background: #f0f7ff;
  border-radius: 8px;
  border-left: 4px solid #409eff;
}

.prediction-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 0;
}

.prediction-label {
  font-size: 12px;
  color: #909399;
}

.prediction-value {
  font-size: 16px;
  font-weight: 600;
  color: #409eff;
}

.prediction-value.confidence {
  color: #67c23a;
}

.prediction-value.model {
  font-size: 13px;
  font-weight: 400;
  color: #2c3e50;
}

.prediction-probs {
  margin-top: 8px;
}

.prediction-probs .prediction-label {
  display: block;
  margin-bottom: 4px;
}

.prob-bars {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.prob-bar-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.prob-label {
  min-width: 50px;
  color: #666;
}

.prob-bar-item .el-progress {
  flex: 1;
}

.prob-value {
  min-width: 44px;
  text-align: right;
  font-weight: 500;
  color: #2c3e50;
}

.chat-messages::-webkit-scrollbar {
  width: 5px;
}
.chat-messages::-webkit-scrollbar-thumb {
  background: #d0d4dc;
  border-radius: 3px;
}
.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-input-area {
  padding: 12px 20px;
  border-top: 1px solid #e4e7ed;
  flex-shrink: 0;
}

.input-wrapper {
  display: flex;
  width: 100%;
}

.chat-input {
  flex: 1;
}

.chat-input :deep(.el-input-group) {
  display: flex;
  width: 100%;
}

.chat-input :deep(.el-input-group__prepend),
.chat-input :deep(.el-input-group__append) {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  padding: 0;
  background: transparent;
  border: none;
}

.chat-input :deep(.el-input__wrapper) {
  flex: 1;
  border-radius: 4px 0 0 4px;
}

.send-btn {
  border-radius: 0 4px 4px 0;
  height: 100%;
  min-height: 40px;
  white-space: nowrap;
  padding: 0 20px;
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}

.input-hint {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  margin-top: 4px;
  font-size: 12px;
  color: #c0c4cc;
  min-height: 20px;
}

.tool-hint {
  color: #409eff;
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>