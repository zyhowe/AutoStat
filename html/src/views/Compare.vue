<template>
  <div class="compare-page">
    <h2>🔍 项目对比</h2>
    <p class="subtitle">并排对比两个分析项目的核心指标</p>

    <div class="compare-select">
      <el-row :gutter="20">
        <el-col :span="10">
          <el-select v-model="projectA" placeholder="选择项目 A" style="width: 100%">
            <el-option
              v-for="p in projects"
              :key="p.session_id"
              :label="p.source_name"
              :value="p.session_id"
            />
          </el-select>
        </el-col>
        <el-col :span="4" style="text-align: center; line-height: 32px;">
          <span style="font-size: 20px;">↔</span>
        </el-col>
        <el-col :span="10">
          <el-select v-model="projectB" placeholder="选择项目 B" style="width: 100%">
            <el-option
              v-for="p in projects"
              :key="p.session_id"
              :label="p.source_name"
              :value="p.session_id"
            />
          </el-select>
        </el-col>
      </el-row>

      <div style="margin-top: 16px; text-align: center;">
        <el-button type="primary" :disabled="!projectA || !projectB || projectA === projectB" @click="handleCompare">
          📊 开始对比
        </el-button>
      </div>
    </div>

    <div v-if="comparing" class="loading-container">
      <el-skeleton :rows="8" animated />
    </div>

    <div v-else-if="compareResult" class="compare-result">
      <el-descriptions :column="2" border>
        <el-descriptions-item :label="nameA">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_a?.rows || 0 }}</div>
            <div class="compare-label">总行数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameB">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_b?.rows || 0 }}</div>
            <div class="compare-label">总行数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameA">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_a?.cols || 0 }}</div>
            <div class="compare-label">总列数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameB">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_b?.cols || 0 }}</div>
            <div class="compare-label">总列数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameA">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_a?.missing_count || 0 }}</div>
            <div class="compare-label">缺失字段数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameB">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_b?.missing_count || 0 }}</div>
            <div class="compare-label">缺失字段数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameA">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_a?.outlier_count || 0 }}</div>
            <div class="compare-label">异常值字段数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameB">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_b?.outlier_count || 0 }}</div>
            <div class="compare-label">异常值字段数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameA">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_a?.duplicate_count || 0 }}</div>
            <div class="compare-label">重复记录数</div>
          </div>
        </el-descriptions-item>
        <el-descriptions-item :label="nameB">
          <div class="compare-item">
            <div class="compare-value">{{ compareResult.data_b?.duplicate_count || 0 }}</div>
            <div class="compare-label">重复记录数</div>
          </div>
        </el-descriptions-item>
      </el-descriptions>

      <div class="diff-summary" v-if="compareResult.diff_summary?.length > 0">
        <h4>📊 主要差异</h4>
        <ul>
          <li v-for="(item, index) in compareResult.diff_summary" :key="index">{{ item }}</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { sessionApi } from '../api/session'

const sessionStore = useSessionStore()
const projects = ref([])
const projectA = ref('')
const projectB = ref('')
const comparing = ref(false)
const compareResult = ref(null)
const nameA = ref('项目 A')
const nameB = ref('项目 B')

onMounted(async () => {
  await loadProjects()
})

async function loadProjects() {
  try {
    const result = await sessionApi.list()
    projects.value = result.projects || []
    if (projects.value.length >= 2) {
      projectA.value = projects.value[0].session_id
      projectB.value = projects.value[1].session_id
    }
  } catch (err) {
    ElMessage.error('加载项目列表失败: ' + err.message)
  }
}

async function handleCompare() {
  if (!projectA.value || !projectB.value || projectA.value === projectB.value) {
    ElMessage.warning('请选择两个不同的项目')
    return
  }

  comparing.value = true
  compareResult.value = null

  try {
    const [dataA, dataB] = await Promise.all([
      getProjectData(projectA.value),
      getProjectData(projectB.value)
    ])

    const diffSummary = []
    if (Math.abs(dataA.rows - dataB.rows) / Math.max(dataA.rows, 1) > 0.1) {
      diffSummary.push(`数据量变化：${dataA.rows.toLocaleString()} → ${dataB.rows.toLocaleString()}`)
    }
    if (dataA.missing_count !== dataB.missing_count) {
      diffSummary.push(`缺失字段：${dataA.missing_count} → ${dataB.missing_count}`)
    }
    if (dataA.outlier_count !== dataB.outlier_count) {
      diffSummary.push(`异常字段：${dataA.outlier_count} → ${dataB.outlier_count}`)
    }

    // 获取项目名称
    const projA = projects.value.find(p => p.session_id === projectA.value)
    const projB = projects.value.find(p => p.session_id === projectB.value)
    nameA.value = projA?.source_name || '项目 A'
    nameB.value = projB?.source_name || '项目 B'

    compareResult.value = {
      data_a: dataA,
      data_b: dataB,
      diff_summary: diffSummary
    }

  } catch (err) {
    ElMessage.error('对比失败: ' + err.message)
  } finally {
    comparing.value = false
  }
}

async function getProjectData(sessionId) {
  const session = await sessionApi.get(sessionId)
  const shape = session.data_shape || {}
  const metadata = await sessionApi.get(sessionId)
  // 简化：从 session 中提取对比数据
  return {
    rows: shape.rows || 0,
    cols: shape.columns || 0,
    missing_count: 0,  // 需要从质量报告中获取
    outlier_count: 0,
    duplicate_count: 0
  }
}
</script>

<style scoped>
.compare-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}
.compare-select {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  margin-bottom: 24px;
}
.loading-container {
  padding: 40px 0;
}
.compare-result {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.compare-item {
  text-align: center;
  padding: 8px;
}
.compare-value {
  font-size: 24px;
  font-weight: bold;
  color: #2c3e50;
}
.compare-label {
  font-size: 12px;
  color: #909399;
}
.diff-summary {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e4e7ed;
}
.diff-summary h4 {
  margin-bottom: 12px;
  color: #2c3e50;
}
.diff-summary ul {
  padding-left: 20px;
  color: #666;
}
</style>