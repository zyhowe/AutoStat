<template>
  <div class="quality-page">
    <h2>📊 数据质量看板</h2>
    <p class="subtitle">五维质量评分，全面了解数据健康状况</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="qualityData" class="quality-content">
      <!-- 综合评分 -->
      <div class="overall-score">
        <el-card class="score-card" shadow="hover">
          <div class="score-number">
            <span class="score-value">{{ qualityData.overall_score }}</span>
            <span class="score-grade">{{ qualityData.grade }}</span>
          </div>
          <div class="score-label">综合评分</div>
        </el-card>
      </div>

      <!-- 五维评分 -->
      <div class="dimensions">
        <el-card v-for="(score, name) in qualityData.dimensions" :key="name" class="dimension-card" shadow="hover">
          <div class="dimension-name">{{ getDimensionLabel(name) }}</div>
          <el-progress
            :percentage="Math.round(score)"
            :color="getProgressColor(score)"
            :stroke-width="12"
          />
          <div class="dimension-value">{{ Math.round(score) }}%</div>
        </el-card>
      </div>

      <!-- 问题清单 -->
      <div class="issues-section">
        <h3>⚠️ 问题清单</h3>
        <el-table :data="issues" border style="width: 100%">
          <el-table-column prop="level" label="级别" width="100">
            <template #default="{ row }">
              <el-tag :type="row.level === 'error' ? 'danger' : 'warning'" size="small">
                {{ row.level === 'error' ? '严重' : '警告' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="field" label="字段" width="150" />
          <el-table-column prop="message" label="问题描述" />
          <el-table-column prop="current" label="当前值" width="120" />
          <el-table-column prop="threshold" label="阈值" width="120" />
        </el-table>
        <div v-if="issues.length === 0" class="empty-text">✅ 未发现问题</div>
      </div>

      <!-- 清洗建议 -->
      <div class="suggestions-section">
        <h3>💡 清洗建议</h3>
        <el-timeline>
          <el-timeline-item
            v-for="(suggestion, index) in suggestions"
            :key="index"
            :type="index === 0 ? 'primary' : 'info'"
          >
            {{ suggestion }}
          </el-timeline-item>
        </el-timeline>
        <div v-if="suggestions.length === 0" class="empty-text">✅ 数据质量良好，无需清洗</div>
      </div>

      <!-- 操作按钮 - 修复底部被遮挡 -->
      <div class="actions" style="margin-top: 30px; padding: 16px 0; border-top: 1px solid #e4e7ed;">
        <el-button type="primary" size="large" @click="goToReport">
          📄 查看分析报告
        </el-button>
        <el-button size="large" @click="goToUpload">
          🔄 重新上传
        </el-button>
      </div>
    </div>

    <div v-else-if="!loading" class="empty-state">
      <el-empty description="暂无质量数据，请先完成分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(true)
const qualityData = ref(null)
const issues = ref([])
const suggestions = ref([])

onMounted(async () => {
  await loadQuality()
})

async function loadQuality() {
  let sessionId = sessionStore.currentSessionId

  // 如果 sessionStore 没有，从 localStorage 获取
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
    console.log('从 localStorage 获取 session_id:', sessionId)
  }

  if (!sessionId) {
    router.push('/')
    return
  }

  // 确保 sessionStore 中有值
  if (!sessionStore.currentSessionId) {
    sessionStore.currentSessionId = sessionId
  }

  loading.value = true
  try {
    const result = await reportApi.getQuality(sessionId)
    qualityData.value = result

    const allIssues = []
    result.alerts?.forEach(alert => {
      if (alert.level === 'error' || alert.level === 'warning') {
        allIssues.push({
          level: alert.level,
          field: alert.field || alert.dimension,
          message: alert.message,
          current: alert.current,
          threshold: alert.threshold
        })
      }
    })
    issues.value = allIssues

    const allSuggestions = []
    result.alerts?.forEach(alert => {
      if (alert.level === 'error') {
        if (alert.dimension === 'completeness') {
          allSuggestions.push(`处理 ${alert.field || '未知字段'} 的缺失值`)
        } else if (alert.dimension === 'accuracy') {
          allSuggestions.push(`检查 ${alert.field || '未知字段'} 的异常值`)
        } else if (alert.dimension === 'consistency') {
          allSuggestions.push('检查数据一致性')
        } else if (alert.dimension === 'uniqueness') {
          allSuggestions.push('去重处理')
        }
      }
    })
    if (allSuggestions.length === 0) {
      allSuggestions.push('数据质量良好，无需清洗')
    }
    suggestions.value = allSuggestions.slice(0, 5)

  } catch (err) {
    ElMessage.error('加载质量报告失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function getDimensionLabel(name) {
  const labels = {
    'completeness': '完整性',
    'accuracy': '准确性',
    'consistency': '一致性',
    'timeliness': '及时性',
    'uniqueness': '唯一性'
  }
  return labels[name] || name
}

function getProgressColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}

function goToReport() {
  router.push('/report')
}

function goToUpload() {
  router.push('/upload')
}
</script>

<style scoped>
.quality-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  padding-bottom: 40px;
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

.overall-score {
  display: flex;
  justify-content: center;
  margin-bottom: 30px;
}
.score-card {
  text-align: center;
  padding: 20px 40px;
  min-width: 200px;
}
.score-number {
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 12px;
}
.score-value {
  font-size: 48px;
  font-weight: bold;
  color: #2c3e50;
}
.score-grade {
  font-size: 24px;
  font-weight: 500;
  padding: 4px 12px;
  border-radius: 4px;
  background: #f0f2f6;
}
.score-label {
  color: #909399;
  margin-top: 8px;
}

.dimensions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 30px;
}
.dimension-card {
  padding: 16px;
}
.dimension-name {
  font-weight: 500;
  margin-bottom: 8px;
  color: #2c3e50;
}
.dimension-value {
  text-align: right;
  margin-top: 4px;
  font-weight: 500;
  font-size: 14px;
  color: #666;
}

.issues-section, .suggestions-section {
  margin-bottom: 30px;
}
.issues-section h3, .suggestions-section h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-text {
  padding: 20px;
  text-align: center;
  color: #67c23a;
}

/* 操作按钮 - 确保可见 */
.actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 30px !important;
  padding: 20px 0 !important;
  border-top: 1px solid #e4e7ed !important;
  position: relative;
  z-index: 10;
}
.actions .el-button {
  min-width: 150px;
}
</style>