<template>
  <div class="report-detail">
    <h2>📄 分析报告</h2>
    <p class="subtitle">{{ getTabLabel() }}</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="15" animated />
    </div>

    <div v-else-if="reportData" class="report-content">
      <!-- 根据 tab 显示不同内容 -->
      <component :is="currentComponent" :reportData="reportData" :qualityData="qualityData" />
    </div>

    <div v-else class="empty-state">
      <el-empty description="暂无报告数据，请先完成分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

// 导入各子组件
import ConclusionsTab from '../components/report/ConclusionsTab.vue'
import InsightsTab from '../components/report/InsightsTab.vue'
import DetailsTab from '../components/report/DetailsTab.vue'
import AuditTab from '../components/report/AuditTab.vue'
import CleaningTab from '../components/report/CleaningTab.vue'

const route = useRoute()
const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(true)
const reportData = ref(null)
const qualityData = ref(null)

const tabComponents = {
  conclusions: ConclusionsTab,
  insights: InsightsTab,
  details: DetailsTab,
  audit: AuditTab,
  cleaning: CleaningTab
}

const currentComponent = computed(() => {
  const tab = route.query.tab || 'conclusions'
  return tabComponents[tab] || ConclusionsTab
})

function getTabLabel() {
  const labels = {
    conclusions: '核心结论',
    insights: '智能解读',
    details: '详细分析',
    audit: '勾稽规则',
    cleaning: '清洗建议'
  }
  return labels[route.query.tab] || '核心结论'
}

async function loadReport() {
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
    const [reportResult, qualityResult] = await Promise.all([
      reportApi.get(sessionId),
      reportApi.getQuality(sessionId)
    ])
    reportData.value = reportResult
    qualityData.value = qualityResult
  } catch (err) {
    ElMessage.error('加载报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function goToUpload() {
  router.push('/upload')
}

onMounted(() => {
  loadReport()
})

watch(() => route.query.tab, () => {
  // tab 切换时重新加载（如果需要）
})
</script>

<style scoped>
.report-detail {
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
</style>