<template>
  <div class="conclusion-solution">
    <h2>💡 结论与方案</h2>
    <p class="subtitle">核心结论、智能解读和清洗建议</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="content">
      <!-- 核心结论 -->
      <div class="section">
        <h3>📊 核心结论</h3>
        <div v-if="conclusions.length === 0" class="empty-tip">
          暂无核心结论
        </div>
        <div class="conclusion-cards" v-else>
          <el-card
            v-for="(conclusion, index) in conclusions"
            :key="index"
            class="conclusion-card"
            shadow="hover"
          >
            <div class="conclusion-icon">{{ conclusion.icon || '📌' }}</div>
            <div class="conclusion-title">{{ conclusion.title }}</div>
            <div class="conclusion-desc">{{ conclusion.description }}</div>
          </el-card>
        </div>
      </div>

      <!-- 智能解读 -->
      <div class="section">
        <h3>💡 智能解读</h3>
        <el-card shadow="hover" v-if="insights.length > 0">
          <el-timeline>
            <el-timeline-item
              v-for="(insight, index) in insights"
              :key="index"
              :type="index === 0 ? 'primary' : 'info'"
            >
              {{ insight }}
            </el-timeline-item>
          </el-timeline>
        </el-card>
        <div v-else class="empty-tip">暂无智能解读</div>
      </div>

      <!-- 清洗建议 -->
      <div class="section">
        <h3>🧹 清洗建议</h3>
        <el-timeline v-if="cleaningSuggestions.length > 0">
          <el-timeline-item
            v-for="(suggestion, index) in cleaningSuggestions"
            :key="index"
            :type="index === 0 ? 'primary' : 'info'"
          >
            {{ suggestion }}
          </el-timeline-item>
        </el-timeline>
        <div v-else class="empty-tip success">
          ✅ 数据质量良好，无需清洗
        </div>
      </div>
    </div>

    <div v-else class="empty-state">
      <el-empty description="请先完成数据分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)

const conclusions = computed(() => {
  return reportData.value?.summary || []
})

const insights = computed(() => {
  return reportData.value?.insights?.findings || []
})

const cleaningSuggestions = computed(() => {
  return reportData.value?.cleaning_suggestions || []
})

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
    const [reportResult, summaryResult, insightsResult] = await Promise.all([
      reportApi.get(sessionId),
      reportApi.getSummary(sessionId),
      reportApi.getInsights(sessionId)
    ])
    reportData.value = {
      ...reportResult,
      summary: summaryResult.conclusions || [],
      insights: insightsResult
    }
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
.conclusion-solution {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
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
.section {
  margin-bottom: 40px;
}
.section h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
}
.empty-tip.success {
  color: #67c23a;
}

.conclusion-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}
.conclusion-card {
  text-align: center;
  padding: 16px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.conclusion-icon {
  font-size: 28px;
  margin-bottom: 8px;
}
.conclusion-title {
  font-weight: bold;
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 4px;
}
.conclusion-desc {
  font-size: 12px;
  color: #909399;
  line-height: 1.4;
}
</style>