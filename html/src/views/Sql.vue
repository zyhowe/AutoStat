<template>
  <div class="sql-page">
    <h2>📝 SQL 生成</h2>
    <p class="subtitle">自然语言描述需求，自动生成 SQL 查询语句</p>

    <el-row :gutter="24">
      <!-- 左侧：推荐问题 -->
      <el-col :span="8">
        <div class="left-panel">
          <div class="section">
            <h4>💡 推荐查询</h4>
            <div class="question-list">
              <div
                v-for="(q, index) in recommendedQuestions"
                :key="index"
                class="question-item"
                @click="sendQuestion(q)"
              >
                <span class="q-icon">📝</span>
                <span class="q-text">{{ q }}</span>
              </div>
            </div>
          </div>

          <div class="section">
            <h4>📊 表信息</h4>
            <div v-if="tableInfo.length === 0" class="empty-tip">
              请先完成数据分析，获取表结构信息
            </div>
            <div v-else class="table-info">
              <div v-for="table in tableInfo" :key="table.name" class="table-item">
                <div class="table-name">{{ table.name }}</div>
                <div class="table-columns">
                  <el-tag
                    v-for="col in table.columns"
                    :key="col"
                    size="small"
                    type="info"
                    class="col-tag"
                  >
                    {{ col }}
                  </el-tag>
                </div>
              </div>
            </div>
          </div>
        </div>
      </el-col>

      <!-- 右侧：对话 -->
      <el-col :span="16">
        <div class="right-panel">
          <div class="chat-header">
            <h4>💬 SQL 对话</h4>
            <el-button size="small" @click="clearChat">清空</el-button>
          </div>

          <div class="chat-messages" ref="chatContainer">
            <div v-if="messages.length === 0" class="empty-chat">
              <el-empty description="从左侧选择问题开始" :image-size="80" />
            </div>
            <div
              v-for="(msg, index) in messages"
              :key="index"
              class="chat-message"
              :class="msg.role"
            >
              <div class="message-avatar">
                {{ msg.role === 'user' ? '👤' : '🤖' }}
              </div>
              <div class="message-content" v-html="formatMessage(msg.content)" />
            </div>
            <div v-if="isStreaming" class="chat-message assistant">
              <div class="message-avatar">🤖</div>
              <div class="message-content streaming" v-html="formatStreamingContent(streamingContent)" />
            </div>
          </div>

          <div class="chat-input">
            <el-input
              v-model="inputQuestion"
              placeholder="描述你的 SQL 需求..."
              @keyup.enter="handleSend"
            >
              <template #append>
                <el-button
                  type="primary"
                  :loading="isStreaming"
                  @click="handleSend"
                >
                  {{ isStreaming ? '生成中' : '发送' }}
                </el-button>
              </template>
            </el-input>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { chatApi } from '../api/chat'

const route = useRoute()
const sessionStore = useSessionStore()

const messages = ref([])
const inputQuestion = ref('')
const isStreaming = ref(false)
const streamingContent = ref('')
const chatContainer = ref(null)
const tableInfo = ref([])

const recommendedQuestions = [
  '查询最近7天的数据',
  '按分类统计数量',
  '关联查询两张表',
  '查询销售额最高的前10条',
  '按月汇总数据',
  '查询某个字段的最大值',
  '统计各分组的平均值',
  '查询数据量最大的表'
]

onMounted(() => {
  loadTableInfo()
  // 如果有 URL 参数问题，自动发送
  if (route.query.question) {
    inputQuestion.value = route.query.question
    handleSend()
  }
})

function loadTableInfo() {
  const session = sessionStore.currentSession
  if (session && session.variable_types) {
    const columns = Object.keys(session.variable_types)
    if (columns.length > 0) {
      tableInfo.value = [{
        name: session.source_name || '数据表',
        columns: columns.slice(0, 20)
      }]
    }
  }
}

function sendQuestion(q) {
  inputQuestion.value = q
  handleSend()
}

async function handleSend() {
  const question = inputQuestion.value.trim()
  if (!question) return
  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  // 添加 SQL 生成提示
  const fullQuestion = `请生成SQL: ${question}`

  messages.value.push({ role: 'user', content: fullQuestion })
  inputQuestion.value = ''
  scrollToBottom()

  isStreaming.value = true
  streamingContent.value = ''

  try {
    await chatApi.chatStream(
      sessionStore.currentSessionId,
      fullQuestion,
      ['json_result'],
      (chunk) => {
        streamingContent.value += chunk
        scrollToBottom()
      },
      () => {
        messages.value.push({ role: 'assistant', content: streamingContent.value })
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

function clearChat() {
  messages.value = []
  streamingContent.value = ''
  isStreaming.value = false
}

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

function formatStreamingContent(content) {
  if (!content) return ''
  return content.replace(/\n/g, '<br>') + '<span class="cursor">▌</span>'
}

function formatMessage(content) {
  if (!content) return ''
  return content
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
}
</script>

<style scoped>
.sql-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.subtitle {
  color: #909399;
  margin-bottom: 24px;
}

.left-panel {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  height: calc(100vh - 160px);
  overflow-y: auto;
}

.section {
  margin-bottom: 20px;
}

.section h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #2c3e50;
  border-bottom: 1px solid #e8ecf1;
  padding-bottom: 8px;
}

.question-list .question-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  margin-bottom: 4px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.question-list .question-item:hover {
  background: #e8ecf1;
}

.question-list .q-icon {
  font-size: 14px;
  flex-shrink: 0;
}

.question-list .q-text {
  font-size: 13px;
  color: #333;
}

.table-info .table-item {
  background: white;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
  border: 1px solid #e4e7ed;
}

.table-name {
  font-weight: 500;
  font-size: 13px;
  color: #2c3e50;
  margin-bottom: 6px;
}

.table-columns {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.col-tag {
  font-size: 11px !important;
  margin: 2px 0;
}

.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
  font-size: 13px;
}

/* 右侧对话 */
.right-panel {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 160px);
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #e4e7ed;
  background: #fafafa;
  flex-shrink: 0;
}

.chat-header h4 {
  margin: 0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.empty-chat {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  align-items: flex-start;
}

.chat-message.user {
  flex-direction: row-reverse;
}

.chat-message.user .message-content {
  background: #409eff;
  color: white;
  border-radius: 12px 12px 4px 12px;
}

.chat-message.assistant .message-content {
  background: #f0f2f6;
  border-radius: 12px 12px 12px 4px;
}

.message-avatar {
  font-size: 20px;
  flex-shrink: 0;
  width: 32px;
  text-align: center;
}

.message-content {
  padding: 10px 14px;
  max-width: 80%;
  word-break: break-word;
  line-height: 1.8;
  font-size: 14px;
}

.message-content.streaming .cursor {
  animation: blink 0.8s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.chat-input {
  padding: 12px 16px;
  border-top: 1px solid #e4e7ed;
  background: #fafafa;
  flex-shrink: 0;
}

.chat-input .el-input {
  width: 100%;
}
</style>