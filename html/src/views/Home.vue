<template>
  <div class="home">
    <div class="welcome-section">
      <h1>📊 AutoStat 智能分析</h1>
      <p class="subtitle">自动识别数据类型、检测数据质量、生成专业分析报告</p>
    </div>

    <div class="actions-section">
      <el-button type="primary" size="large" @click="handleNewAnalysis">
        <el-icon><Plus /></el-icon>
        新建分析
      </el-button>
    </div>

    <div class="recent-section">
      <h3>📋 最近项目</h3>
      <el-table :data="projects" style="width: 100%" v-loading="loading">
        <el-table-column prop="source_name" label="项目名称" />
        <el-table-column prop="analysis_type" label="类型" width="120">
          <template #default="{ row }">
            <el-tag size="small">{{ row.analysis_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="200" />
        <el-table-column prop="data_shape" label="数据规模" width="150">
          <template #default="{ row }">
            {{ row.data_shape?.rows || 0 }} 行 × {{ row.data_shape?.columns || 0 }} 列
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small" type="primary" @click="handleOpen(row.session_id)">打开</el-button>
            <el-button size="small" type="danger" @click="handleDelete(row.session_id)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div v-if="projects.length === 0" class="empty-state">
        <el-empty description="暂无项目，点击「新建分析」开始" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useSessionStore } from '../stores/session'

const router = useRouter()
const sessionStore = useSessionStore()
const projects = ref([])
const loading = ref(false)

onMounted(async () => {
  await loadProjects()
})

async function loadProjects() {
  loading.value = true
  try {
    await sessionStore.loadProjects()
    projects.value = sessionStore.projects
  } catch (err) {
    ElMessage.error('加载项目失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function handleNewAnalysis() {
  router.push('/upload')
}

function handleOpen(sessionId) {
  sessionStore.currentSessionId = sessionId
  router.push('/quality')
}

async function handleDelete(sessionId) {
  try {
    await ElMessageBox.confirm('确定要删除这个项目吗？此操作不可恢复。', '确认删除', {
      type: 'warning'
    })
    await sessionStore.deleteSession(sessionId)
    await loadProjects()
    ElMessage.success('项目已删除')
  } catch (err) {
    if (err !== 'cancel') {
      ElMessage.error('删除失败: ' + err.message)
    }
  }
}
</script>

<style scoped>
.home {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.welcome-section {
  text-align: center;
  padding: 40px 0;
}
.welcome-section h1 {
  font-size: 32px;
  color: #2c3e50;
  margin-bottom: 12px;
}
.subtitle {
  font-size: 16px;
  color: #909399;
}
.actions-section {
  text-align: center;
  margin-bottom: 40px;
}
.recent-section h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-state {
  padding: 40px 0;
}
</style>