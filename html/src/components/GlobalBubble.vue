<template>
  <div class="global-bubble" ref="bubbleRef">
    <!-- 气泡按钮 -->
    <div class="bubble-btn" @click.stop="togglePanel" :class="{ active: panelVisible }">
      <el-icon v-if="!panelVisible"><ChatDotRound /></el-icon>
      <el-icon v-else><Close /></el-icon>
      <span class="bubble-tip">问AI</span>
    </div>

    <!-- 气泡面板 -->
    <transition name="slide-up">
      <div v-if="panelVisible" class="bubble-panel" @click.stop>
        <!-- ====== 上半部分：对话框 ====== -->
        <div class="panel-chat">
          <div class="chat-messages" ref="chatContainer">
            <div v-if="chatMessages.length === 0" class="chat-empty">
              <span>💬 输入问题，AI 帮你解答</span>
            </div>
            <div
              v-for="(msg, index) in chatMessages"
              :key="index"
              class="chat-msg"
              :class="msg.role"
            >
              <span class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</span>
              <span class="msg-content">{{ msg.content }}</span>
            </div>
            <div v-if="isStreaming" class="chat-msg assistant">
              <span class="msg-avatar">🤖</span>
              <span class="msg-content">{{ streamingContent }}<span class="cursor">▌</span></span>
            </div>
          </div>
          <div class="chat-input">
            <el-input
              v-model="inputQuestion"
              size="small"
              placeholder="输入问题..."
              @keyup.enter="handleSend"
            >
              <template #append>
                <el-button size="small" type="primary" :loading="isStreaming" @click="handleSend">
                  发送
                </el-button>
              </template>
            </el-input>
          </div>
        </div>

        <!-- ====== 下半部分：推荐问题列表 ====== -->
        <div class="panel-questions">
          <div class="questions-header">
            <span>💡 推荐问题</span>
          </div>
          <div class="questions-list">
            <div
              v-for="(q, index) in currentQuestions"
              :key="index"
              class="question-item"
              @click="handleQuestionClick(q)"
            >
              <span class="q-icon">{{ q.icon || '💬' }}</span>
              <span class="q-text">{{ q.text }}</span>
            </div>
            <div v-if="currentQuestions.length === 0" class="empty-tip">
              当前页面暂无推荐问题
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ChatDotRound, Close } from '@element-plus/icons-vue'
import { useSessionStore } from '../stores/session'
import { chatApi } from '../api/chat'

const router = useRouter()
const route = useRoute()
const sessionStore = useSessionStore()
const bubbleRef = ref(null)
const chatContainer = ref(null)

const panelVisible = ref(false)
const inputQuestion = ref('')
const isStreaming = ref(false)
const streamingContent = ref('')
const chatMessages = ref([])

// ==================== 各页面推荐问题映射 ====================
const questionMap = {
  '/quality': [
    { icon: '📊', text: '解读质量评分，说明各项得分的含义' },
    { icon: '🧹', text: '给出数据清洗建议，按优先级排序' },
    { icon: '🚨', text: '识别当前数据的主要质量问题' }
  ],
  '/report': [
    { icon: '🎯', text: '总结分析报告的核心结论' },
    { icon: '💡', text: '解读关键发现和数据洞察' },
    { icon: '🔗', text: '解释勾稽规则的含义和业务价值' }
  ],
  '/upload': [
    { icon: '📋', text: '如何选择字段类型？' },
    { icon: '🔍', text: '数据预览有什么问题需要注意？' }
  ],
  '/models': [
    { icon: '🤖', text: '根据当前数据推荐什么模型？' },
    { icon: '📊', text: '解释模型训练结果和评估指标' }
  ],
  '/sql': [
    { icon: '📝', text: '生成查询最近7天数据的SQL' },
    { icon: '📊', text: '生成按分类统计数量的SQL' },
    { icon: '🔗', text: '生成关联查询两张表的SQL' }
  ],
  '/natural': [
    { icon: '🔍', text: '查询销售额最高的前10条记录' },
    { icon: '📊', text: '统计各分类的分布情况' },
    { icon: '📈', text: '分析各月的变化趋势' }
  ],
  '/ai': [
    { icon: '💬', text: '解读数据的主要特征和业务含义' },
    { icon: '⚠️', text: '分析数据质量问题并给出清洗建议' }
  ],
  '/compare': [
    { icon: '📊', text: '对比两个项目的差异和共同点' }
  ],
  '/settings': [
    { icon: '⚙️', text: '如何配置大模型API？' }
  ]
}

const defaultQuestions = [
  { icon: '💬', text: '如何使用这个平台？' },
  { icon: '📊', text: '数据上传支持哪些格式？' }
]

const currentQuestions = computed(() => {
  const path = route.path
  if (questionMap[path]) return questionMap[path]
  for (const [key, questions] of Object.entries(questionMap)) {
    if (path.startsWith(key) && key !== '/') return questions
  }
  return defaultQuestions
})

// ==================== 发送消息 ====================
async function handleSend() {
  const question = inputQuestion.value.trim()
  if (!question) return

  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  chatMessages.value.push({ role: 'user', content: question })
  inputQuestion.value = ''
  scrollToBottom()

  isStreaming.value = true
  streamingContent.value = ''

  try {
    await chatApi.chatStream(
      sessionStore.currentSessionId,
      question,
      ['json_result'],
      (chunk) => {
        streamingContent.value += chunk
        scrollToBottom()
      },
      () => {
        chatMessages.value.push({ role: 'assistant', content: streamingContent.value })
        streamingContent.value = ''
        isStreaming.value = false
        scrollToBottom()
      },
      (error) => {
        ElMessage.error('对话失败: ' + error)
        isStreaming.value = false
      }
    )
  } catch (err) {
    ElMessage.error('对话失败: ' + err.message)
    isStreaming.value = false
  }
}

function handleQuestionClick(q) {
  inputQuestion.value = q.text
  handleSend()
}

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

function togglePanel() {
  panelVisible.value = !panelVisible.value
}

function handleClickOutside(event) {
  if (bubbleRef.value && !bubbleRef.value.contains(event.target)) {
    panelVisible.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.global-bubble {
  position: fixed;
  bottom: 40px;
  right: 40px;
  z-index: 9999;
}

.bubble-btn {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
  transition: all 0.3s ease;
  position: relative;
}

.bubble-btn:hover {
  transform: scale(1.08);
  box-shadow: 0 6px 30px rgba(102, 126, 234, 0.5);
}
.bubble-btn .el-icon { font-size: 28px; }
.bubble-btn.active { background: #f56c6c; }

.bubble-tip {
  position: absolute;
  top: -8px;
  right: -8px;
  background: #ff6b6b;
  color: white;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 10px;
  font-weight: bold;
}

/* 面板 */
.bubble-panel {
  position: absolute;
  bottom: 75px;
  right: 0;
  width: 400px;
  max-height: 520px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ===== 上半部分：对话框 ===== */
.panel-chat {
  border-bottom: 1px solid #f0f0f0;
  display: flex;
  flex-direction: column;
  max-height: 260px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 10px 14px;
  min-height: 60px;
  max-height: 200px;
}

.chat-empty {
  text-align: center;
  color: #bbb;
  font-size: 13px;
  padding: 20px 0;
}

.chat-msg {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 13px;
  align-items: flex-start;
}

.chat-msg.user {
  flex-direction: row-reverse;
}
.chat-msg.user .msg-content {
  background: #409eff;
  color: white;
  padding: 6px 12px;
  border-radius: 12px 12px 4px 12px;
  max-width: 75%;
}
.chat-msg.assistant .msg-content {
  background: #f0f2f6;
  padding: 6px 12px;
  border-radius: 12px 12px 12px 4px;
  max-width: 75%;
  word-break: break-word;
}
.msg-avatar { font-size: 16px; flex-shrink: 0; }
.cursor { animation: blink 0.8s infinite; }
@keyframes blink { 0%,50% { opacity: 1; } 51%,100% { opacity: 0; } }

.chat-input {
  padding: 6px 12px 10px 12px;
  flex-shrink: 0;
}
.chat-input .el-input { width: 100%; }

/* ===== 下半部分：推荐问题 ===== */
.panel-questions {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.questions-header {
  padding: 10px 16px 6px 16px;
  font-size: 13px;
  font-weight: 600;
  color: #333;
  flex-shrink: 0;
  border-bottom: 1px solid #f5f5f5;
}

.questions-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
  max-height: 180px;
}

.question-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px;
  cursor: pointer;
  transition: background 0.15s;
  border-bottom: 1px solid #f8f8f8;
}
.question-item:hover { background: #f0f2ff; }
.q-icon { font-size: 14px; flex-shrink: 0; }
.q-text { font-size: 12px; color: #333; line-height: 1.4; }

.empty-tip {
  padding: 20px;
  text-align: center;
  color: #999;
  font-size: 13px;
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

@media (max-width: 768px) {
  .bubble-btn { width: 50px; height: 50px; }
  .bubble-btn .el-icon { font-size: 22px; }
  .bubble-panel { width: 310px; right: -10px; max-height: 460px; }
}
</style>