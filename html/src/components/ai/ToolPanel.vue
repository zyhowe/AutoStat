<template>
  <div class="tool-panel">
    <!-- ===== 推荐问题 ===== -->
    <div class="section">
      <div class="section-header">
        <span class="section-title">💡 推荐问题</span>
        <el-button size="small" text @click="refreshQuestions" :loading="loadingQuestions">
          🔄
        </el-button>
      </div>

      <div v-if="loadingQuestions" class="loading-skeleton">
        <el-skeleton :rows="5" animated />
      </div>

      <div v-else-if="!hasAnyQuestions" class="empty-text">
        暂无推荐问题，请先完成数据分析
      </div>

      <div v-else class="question-tree">
        <div
          v-for="group in questionGroups"
          :key="group.key"
          class="question-group"
          :class="{ active: activeGroup === group.key }"
        >
          <div class="group-header" @click="toggleGroup(group.key)">
            <span class="group-icon">{{ group.icon }}</span>
            <span class="group-title">{{ group.label }}</span>
            <span class="group-toggle">{{ expandedGroups[group.key] ? '▼' : '▶' }}</span>
          </div>

          <div v-show="expandedGroups[group.key]" class="sub-group-list">
            <div
              v-for="sub in group.subItems"
              :key="sub.toolId"
              class="sub-group"
            >
              <div class="sub-header">
                <span class="sub-icon">{{ sub.icon }}</span>
                <span class="sub-title">{{ sub.label }}</span>
                <span v-if="getSubQuestionCount(sub.toolId) === 0" class="sub-empty">暂无推荐</span>
              </div>

              <div v-if="getSubQuestionCount(sub.toolId) > 0" class="question-list">
                <!-- 个性化推荐（蓝色） -->
                <div
                  v-for="(q, idx) in getPersonalizedQuestions(sub.toolId)"
                  :key="'p_' + idx"
                  class="question-item personalized"
                  @click="onQuestionClick(q, sub.dataKey)"
                >
                  <span class="q-icon">{{ q.icon || '💡' }}</span>
                  <span class="q-text personalized-text">{{ q.text }}</span>
                </div>

                <!-- 共性推荐（灰色） -->
                <div
                  v-for="(q, idx) in getCommonQuestions(sub.toolId)"
                  :key="'c_' + idx"
                  class="question-item common"
                  @click="onQuestionClick(q, sub.dataKey)"
                >
                  <span class="q-icon">{{ q.icon || '💡' }}</span>
                  <span class="q-text common-text">{{ q.text }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import commonQuestionsData from '../../data/common_questions.json'

const props = defineProps({
  scene: {
    type: String,
    default: 'general'
  },
  sessionId: {
    type: String,
    default: ''
  },
  personalizedQuestions: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['question-click', 'refresh-questions'])

const loadingQuestions = ref(false)

const expandedGroups = ref({
  explore: true,
  quality: true,
  query: true,
  predict: true,
  report: true
})

const activeGroup = ref('')

// ===== 大项配置（含 dataKey 映射） =====
const questionGroups = [
  {
    key: 'explore',
    icon: '📈',
    label: '探索分析',
    subItems: [
      { toolId: 'describe_distribution', icon: '📊', label: '描述字段分布', dataKey: 'data_overview.distribution' },
      { toolId: 'analyze_correlation', icon: '🔗', label: '分析变量相关性', dataKey: 'pattern_discovery.correlation' },
      { toolId: 'detect_timeseries', icon: '📈', label: '检测时间序列规律', dataKey: 'pattern_discovery.timeseries' },
      { toolId: 'identify_categorical', icon: '🏷️', label: '识别分类特征', dataKey: 'data_overview.categorical' }
    ]
  },
  {
    key: 'quality',
    icon: '✅',
    label: '质量诊断',
    subItems: [
      { toolId: 'interpret_quality', icon: '📊', label: '解读质量评分', dataKey: 'quality.overall' },
      { toolId: 'check_outliers', icon: '🚨', label: '检查异常与缺失', dataKey: 'data_validation.outliers' },
      { toolId: 'validate_rules', icon: '🔗', label: '验证勾稽规则', dataKey: 'data_validation.audit_rules' },
      { toolId: 'detect_duplicates', icon: '📋', label: '检测重复记录', dataKey: 'data_validation.duplicates' }
    ]
  },
  {
    key: 'query',
    icon: '🔍',
    label: '数据查询',
    subItems: [
      { toolId: 'natural_query', icon: '🔍', label: '自然语言查数', dataKey: 'data_overview.natural_query' },
      { toolId: 'generate_sql', icon: '📝', label: '生成 SQL', dataKey: 'data_overview.generate_sql' }
    ]
  },
  {
    key: 'predict',
    icon: '🤖',
    label: '智能预测',
    subItems: [
      { toolId: 'recommend_model', icon: '🤖', label: '推荐建模方案', dataKey: 'smart_prediction.model_recommend' },
      { toolId: 'execute_predict', icon: '🔮', label: '执行预测', dataKey: 'smart_prediction.forecast' }
    ]
  },
  {
    key: 'report',
    icon: '📋',
    label: '报告摘要',
    subItems: [
      { toolId: 'generate_conclusions', icon: '📋', label: '生成核心结论', dataKey: 'report_summary.conclusions' },
      { toolId: 'extract_insights', icon: '💡', label: '提炼业务洞察', dataKey: 'report_summary.insights' }
    ]
  }
]

// ===== 小项 → 场景/子项映射（用于获取个性化推荐） =====
const subToSceneMap = {
  'describe_distribution': { scene: 'data_overview', sub: 'distribution' },
  'analyze_correlation': { scene: 'pattern_discovery', sub: 'correlation' },
  'detect_timeseries': { scene: 'pattern_discovery', sub: 'timeseries' },
  'identify_categorical': { scene: 'data_overview', sub: 'categorical' },
  'interpret_quality': { scene: 'quality', sub: 'overall' },
  'check_outliers': { scene: 'data_validation', sub: 'outliers' },
  'validate_rules': { scene: 'data_validation', sub: 'audit_rules' },
  'detect_duplicates': { scene: 'data_validation', sub: 'duplicates' },
  'natural_query': { scene: 'data_overview', sub: 'natural_query' },
  'generate_sql': { scene: 'data_overview', sub: 'generate_sql' },
  'recommend_model': { scene: 'smart_prediction', sub: 'model_recommend' },
  'execute_predict': { scene: 'smart_prediction', sub: 'forecast' },
  'generate_conclusions': { scene: 'report_summary', sub: 'conclusions' },
  'extract_insights': { scene: 'report_summary', sub: 'insights' }
}

// ===== 计算属性 =====
const hasAnyQuestions = computed(() => {
  const toolIds = [
    'describe_distribution', 'analyze_correlation', 'detect_timeseries', 'identify_categorical',
    'interpret_quality', 'check_outliers', 'validate_rules', 'detect_duplicates',
    'natural_query', 'generate_sql',
    'recommend_model', 'execute_predict',
    'generate_conclusions', 'extract_insights'
  ]
  return toolIds.some(id => getSubQuestionCount(id) > 0)
})

function getSubQuestionCount(toolId) {
  const personalized = getPersonalizedQuestions(toolId)
  const common = getCommonQuestions(toolId)
  return personalized.length + common.length
}

function getPersonalizedQuestions(toolId) {
  const map = subToSceneMap[toolId]
  if (!map) return []
  const sceneData = props.personalizedQuestions[map.scene]
  if (!sceneData) return []
  return sceneData[map.sub] || []
}

function getCommonQuestions(toolId) {
  const map = subToSceneMap[toolId]
  if (!map) return []
  const sceneData = commonQuestionsData[map.scene]
  if (!sceneData) return []
  return sceneData[map.sub] || []
}

function toggleGroup(groupKey) {
  expandedGroups.value[groupKey] = !expandedGroups.value[groupKey]
  try {
    localStorage.setItem('ai_recommend_groups', JSON.stringify(expandedGroups.value))
  } catch (e) { /* ignore */ }
}

// ===== 点击问题：传递问题文本 + dataKey =====
function onQuestionClick(q, dataKey) {
  emit('question-click', {
    text: q.text,
    dataKey: dataKey
  })
}

function refreshQuestions() {
  emit('refresh-questions')
}

onMounted(() => {
  try {
    const saved = localStorage.getItem('ai_recommend_groups')
    if (saved) {
      const parsed = JSON.parse(saved)
      expandedGroups.value = { ...expandedGroups.value, ...parsed }
    }
  } catch (e) { /* ignore */ }
})

defineExpose({ refreshQuestions })
</script>

<style scoped>
.tool-panel {
  padding: 12px 16px;
  height: 100%;
  overflow-y: auto;
  background: #f8f9fa;
  border-right: 1px solid #e4e7ed;
}

.section {
  margin-bottom: 8px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}

.loading-skeleton {
  padding: 8px 0;
}

.empty-text {
  padding: 12px 0;
  text-align: center;
  color: #bbb;
  font-size: 13px;
}

.question-tree {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.question-group {
  border-radius: 6px;
  border: 1px solid transparent;
  transition: border-color 0.2s;
}

.question-group.active {
  border-color: #409eff;
  background: #f0f7ff;
}

.group-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.2s;
  user-select: none;
}

.group-header:hover {
  background: #e8ecf1;
}

.group-icon {
  font-size: 14px;
}

.group-title {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
}

.group-toggle {
  font-size: 11px;
  color: #909399;
}

.sub-group-list {
  padding: 2px 0 4px 10px;
}

.sub-group {
  margin-bottom: 4px;
}

.sub-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  font-size: 12px;
  color: #666;
  border-bottom: 1px dashed #eee;
}

.sub-icon {
  font-size: 12px;
}

.sub-title {
  font-weight: 500;
  color: #555;
}

.sub-empty {
  font-size: 11px;
  color: #bbb;
  margin-left: auto;
}

.question-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px 0 4px 20px;
}

.question-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.15s;
  font-size: 12px;
  position: relative;
}

.question-item:hover {
  background: #e8ecf1;
}

.question-item.personalized {
  border-left: 2px solid #409eff;
  padding-left: 10px;
}
.question-item.personalized .personalized-text {
  color: #409eff;
}

.question-item.common {
  border-left: 2px solid #c0c4cc;
  padding-left: 10px;
}
.question-item.common .common-text {
  color: #606266;
}

.q-icon {
  font-size: 13px;
  flex-shrink: 0;
}

.q-text {
  flex: 1;
  line-height: 1.4;
}

.tool-panel::-webkit-scrollbar {
  width: 4px;
}
.tool-panel::-webkit-scrollbar-thumb {
  background: #d0d4dc;
  border-radius: 2px;
}
.tool-panel::-webkit-scrollbar-track {
  background: transparent;
}
</style>