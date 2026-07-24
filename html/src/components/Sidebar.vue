<template>
  <div class="sidebar">
    <!-- 菜单区域 -->
    <div class="sidebar-menu-wrapper" ref="menuWrapperRef">
      <el-menu
        :default-active="activeMenu"
        router
        class="sidebar-menu"
        @select="handleMenuSelect"
      >
        <el-menu-item index="/upload">
          <el-icon><Upload /></el-icon>
          <span>上传数据</span>
        </el-menu-item>

        <el-menu-item index="/report-summary">
          <el-icon><Document /></el-icon>
          <span>报告总览</span>
        </el-menu-item>

        <el-menu-item index="/data-overview">
          <el-icon><DataAnalysis /></el-icon>
          <span>数据概览</span>
        </el-menu-item>

        <el-menu-item index="/quality">
          <el-icon><Monitor /></el-icon>
          <span>质量看板</span>
        </el-menu-item>

        <el-menu-item index="/data-validation">
          <el-icon><CircleCheck /></el-icon>
          <span>数据核验</span>
        </el-menu-item>

        <el-menu-item index="/pattern-discovery">
          <el-icon><TrendCharts /></el-icon>
          <span>规律发现</span>
        </el-menu-item>

        <el-menu-item index="/models">
          <el-icon><Cpu /></el-icon>
          <span>智能预测</span>
        </el-menu-item>

        <el-menu-item index="/scenario-analysis">
          <el-icon><MagicStick /></el-icon>
          <span>场景分析</span>
        </el-menu-item>

        <el-menu-item index="/ai">
          <el-icon><ChatDotRound /></el-icon>
          <span>AI助手</span>
        </el-menu-item>

        <el-menu-item index="/settings">
          <el-icon><Setting /></el-icon>
          <span>设置</span>
        </el-menu-item>
      </el-menu>
    </div>

    <!-- 可拖动分隔条 -->
    <div
      class="sidebar-divider"
      @mousedown="startDrag"
      :style="{ cursor: isDragging ? 'row-resize' : 'ns-resize' }"
    >
      <div class="divider-line"></div>
    </div>

    <!-- 项目列表区域（可滚动） -->
    <div class="projects-section" ref="projectsSectionRef">
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
            @click.stop="deleteProject(project.session_id, project.source_name)"
            title="删除项目"
          >
            🗑️
          </el-button>
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
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
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

// 拖拽分隔条
const projectsSectionRef = ref(null)
const menuWrapperRef = ref(null)
const isDragging = ref(false)
const startY = ref(0)
const startHeight = ref(0)
const minProjectsHeight = 120
const maxProjectsHeight = 500

// ===== 拖拽逻辑 =====
function startDrag(e) {
  isDragging.value = true
  startY.value = e.clientY

  const section = projectsSectionRef.value
  if (section) {
    // 获取当前实际高度（可能是 flex 计算后的高度）
    startHeight.value = section.getBoundingClientRect().height
  }

  document.addEventListener('mousemove', onDrag)
  document.addEventListener('mouseup', stopDrag)
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'row-resize'
}

function onDrag(e) {
  if (!isDragging.value) return

  const section = projectsSectionRef.value
  if (!section) return

  // 计算新高度：鼠标移动距离（向上拖拽增加高度）
  const deltaY = startY.value - e.clientY
  let newHeight = startHeight.value + deltaY
  newHeight = Math.max(minProjectsHeight, Math.min(maxProjectsHeight, newHeight))

  // 使用 flex-basis 控制高度，避免与 flex:1 冲突
  section.style.flex = `0 0 ${newHeight}px`
  section.style.height = `${newHeight}px`
}

function stopDrag() {
  isDragging.value = false
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
}

// ===== 项目列表 =====
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
  overflow: hidden;
}

/* ===== 菜单区域 ===== */
.sidebar-menu-wrapper {
  flex: 1 1 auto;
  overflow: hidden;
  min-height: 0;
}

.sidebar-menu {
  border-right: none;
  background: transparent;
  height: 100%;
  overflow-y: auto;
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

/* 菜单滚动条 */
.sidebar-menu::-webkit-scrollbar {
  width: 4px;
}
.sidebar-menu::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 2px;
}
.sidebar-menu::-webkit-scrollbar-track {
  background: transparent;
}

/* ===== 分隔条 ===== */
.sidebar-divider {
  flex: 0 0 6px;
  background: #e4e7ed;
  cursor: ns-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
  position: relative;
  z-index: 10;
  min-height: 6px;
  flex-shrink: 0;
}

.sidebar-divider:hover {
  background: #409eff;
}

.divider-line {
  width: 30px;
  height: 2px;
  background: #c0c4cc;
  border-radius: 1px;
  transition: background 0.2s;
}

.sidebar-divider:hover .divider-line {
  background: #fff;
}

/* ===== 项目区域 ===== */
.projects-section {
  flex: 0 0 200px; /* 默认高度，拖拽时会被覆盖 */
  min-height: 120px;
  max-height: 500px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #f5f7fa;
  transition: none; /* 拖拽时实时响应，不加过渡 */
}

.projects-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px 6px 16px;
  flex-shrink: 0;
}

.projects-title {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
}

.projects-list {
  flex: 1 1 0;
  overflow-y: auto;
  padding: 0 8px 4px 8px;
  min-height: 0;
}

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

/* ===== 底部状态 ===== */
.sidebar-footer {
  flex-shrink: 0;
  padding: 6px 16px 10px 16px;
}
.sidebar-footer .el-divider {
  margin: 0 0 6px 0;
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