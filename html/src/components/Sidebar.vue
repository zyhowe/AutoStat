<template>
  <div class="sidebar">
    <el-menu
      :default-active="activeMenu"
      router
      class="sidebar-menu"
      @select="handleMenuSelect"
    >
      <!-- 上传数据 -->
      <el-menu-item index="/upload">
        <el-icon><Upload /></el-icon>
        <span>上传数据</span>
      </el-menu-item>

      <!-- 报告总览 -->
      <el-menu-item index="/report-summary">
        <el-icon><Document /></el-icon>
        <span>报告总览</span>
      </el-menu-item>

      <!-- 数据概览 -->
      <el-menu-item index="/data-overview">
        <el-icon><DataAnalysis /></el-icon>
        <span>数据概览</span>
      </el-menu-item>

      <!-- 质量看板 -->
      <el-menu-item index="/quality">
        <el-icon><Monitor /></el-icon>
        <span>质量看板</span>
      </el-menu-item>

      <!-- 数据核验 -->
      <el-menu-item index="/data-validation">
        <el-icon><CircleCheck /></el-icon>
        <span>数据核验</span>
      </el-menu-item>

      <!-- 规律发现 -->
      <el-menu-item index="/pattern-discovery">
        <el-icon><TrendCharts /></el-icon>
        <span>规律发现</span>
      </el-menu-item>

      <!-- 智能预测 -->
      <el-menu-item index="/models">
        <el-icon><Cpu /></el-icon>
        <span>智能预测</span>
      </el-menu-item>

      <!-- 场景分析（合并原场景推导和仪表板） -->
      <el-menu-item index="/scenario-analysis">
        <el-icon><MagicStick /></el-icon>
        <span>场景分析</span>
      </el-menu-item>

      <!-- AI助手 -->
      <el-menu-item index="/ai">
        <el-icon><ChatDotRound /></el-icon>
        <span>AI助手</span>
      </el-menu-item>

      <!-- 设置 -->
      <el-menu-item index="/settings">
        <el-icon><Setting /></el-icon>
        <span>设置</span>
      </el-menu-item>
    </el-menu>

    <!-- 最近项目 -->
    <div class="projects-section">
      <el-divider />
      <div class="projects-header">
        <span class="projects-title">📋 最近项目</span>
        <el-button size="small" type="primary" text @click="loadProjects">🔄</el-button>
      </div>
      <div v-if="loadingProjects" class="projects-loading">
        <el-skeleton :rows="3" animated />
      </div>
      <div v-else-if="projects.length === 0" class="projects-empty">
        <span style="font-size: 12px; color: #909399;">暂无项目</span>
      </div>
      <div v-else class="projects-list">
        <div
          v-for="project in projects"
          :key="project.session_id"
          class="project-item"
          :class="{ active: project.session_id === sessionStore.currentSessionId }"
        >
          <div class="project-info" @click="loadProject(project.session_id)">
            <div class="project-name">{{ project.source_name }}</div>
            <div class="project-meta">
              {{ project.data_shape?.rows || 0 }}行 × {{ project.data_shape?.columns || 0 }}列
            </div>
          </div>
          <el-button
            size="small"
            type="danger"
            text
            @click="deleteProject(project.session_id, project.source_name)"
            title="删除项目"
          >
            🗑️
          </el-button>
        </div>
      </div>
    </div>

    <!-- 底部状态 -->
    <div class="sidebar-footer">
      <el-divider />
      <div class="status-info">
        <div class="status-item">
          <span class="status-dot" :class="sessionStore.hasSession ? 'online' : 'offline'"></span>
          <span class="status-text">{{ sessionStore.hasSession ? '已加载项目' : '未加载项目' }}</span>
        </div>
        <div v-if="sessionStore.hasSession" class="status-item">
          <span class="status-label">项目:</span>
          <span class="status-text project-name">{{ sessionStore.sessionName }}</span>
        </div>
        <div v-if="sessionStore.currentSession" class="status-item">
          <span class="status-label">数据:</span>
          <span class="status-text">
            {{ sessionStore.currentSession.data_shape?.rows || 0 }} 行
            × {{ sessionStore.currentSession.data_shape?.columns || 0 }} 列
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { sessionApi } from '../api/session'

import {
  Upload,
  DataAnalysis,
  Monitor,
  CircleCheck,
  TrendCharts,
  Document,
  Cpu,
  ChatDotRound,
  Setting,
  MagicStick
} from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const sessionStore = useSessionStore()

const activeMenu = computed(() => route.path)
const projects = ref([])
const loadingProjects = ref(false)

onMounted(async () => {
  await loadProjects()
})

watch(() => sessionStore.currentSessionId, async (newId, oldId) => {
  if (newId !== oldId) {
    await loadProjects()
  }
})

function handleMenuSelect(index) {
  if (index.includes('?')) {
    router.push(index)
  }
}

async function loadProjects() {
  loadingProjects.value = true
  try {
    const result = await sessionApi.list()
    projects.value = result.projects || []
  } catch (err) {
    console.error('加载项目列表失败:', err)
  } finally {
    loadingProjects.value = false
  }
}

async function loadProject(sessionId) {
  if (sessionId === sessionStore.currentSessionId) {
    router.go(0)
    return
  }
  try {
    await sessionStore.loadSession(sessionId)
    ElMessage.success('已加载项目')
    router.replace('/report-summary')
  } catch (err) {
    ElMessage.error('加载项目失败: ' + err.message)
  }
}

async function deleteProject(sessionId, sourceName) {
  try {
    await ElMessageBox.confirm(
      `确定要删除项目「${sourceName}」吗？此操作不可恢复。`,
      '确认删除',
      { type: 'warning' }
    )
    await sessionApi.delete(sessionId)
    ElMessage.success('项目已删除')
    if (sessionStore.currentSessionId === sessionId) {
      sessionStore.currentSessionId = null
      sessionStore.currentSession = null
    }
    await loadProjects()
  } catch (err) {
    if (err !== 'cancel') {
      ElMessage.error('删除失败: ' + err.message)
    }
  }
}
</script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #f5f7fa;
  border-right: 1px solid #e4e7ed;
  overflow: hidden; /* 防止整体滚动 */
}

.sidebar-menu {
  flex-shrink: 0; /* 菜单不收缩 */
  border-right: none;
  background: transparent;
}
.sidebar-menu .el-menu-item.is-active {
  background-color: #ecf5ff !important;
  border-right: 3px solid #409eff;
}
.sidebar-menu .el-menu-item:hover {
  background-color: #ecf5ff !important;
}
.sidebar-menu .el-menu-item {
  padding-left: 20px !important;
  font-size: 14px;
}

.projects-section {
  flex: 1 1 0; /* 占据剩余空间，允许收缩 */
  min-height: 0; /* 允许 flex 子项收缩 */
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.projects-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 16px 8px 16px;
  flex-shrink: 0;
}
.projects-title {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
}
.projects-list {
  flex: 1 1 0;
  overflow-y: auto; /* 垂直滚动 */
  padding: 0 8px;
  min-height: 0;
}
/* 自定义滚动条 */
.projects-list::-webkit-scrollbar {
  width: 4px;
}
.projects-list::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 2px;
}
.projects-list::-webkit-scrollbar-track {
  background: transparent;
}

.projects-loading {
  padding: 8px 16px;
  flex-shrink: 0;
}
.projects-empty {
  padding: 16px;
  text-align: center;
  flex-shrink: 0;
}

.project-item {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  margin-bottom: 2px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}
.project-item:hover {
  background: #e8ecf1;
}
.project-item.active {
  background: #ecf5ff;
}
.project-item .project-info {
  flex: 1;
  min-width: 0;
  cursor: pointer;
}
.project-item .project-name {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.project-item .project-meta {
  font-size: 11px;
  color: #909399;
}
.project-item .el-button {
  flex-shrink: 0;
}

.sidebar-footer {
  flex-shrink: 0; /* 底部不收缩 */
  padding: 8px 16px 12px 16px;
}
.sidebar-footer .el-divider {
  margin: 0 0 8px 0;
}
.status-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.status-dot.online {
  background: #67c23a;
}
.status-dot.offline {
  background: #909399;
}
.status-label {
  color: #909399;
  flex-shrink: 0;
}
.status-text {
  color: #2c3e50;
}
.status-text.project-name {
  font-weight: 500;
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>