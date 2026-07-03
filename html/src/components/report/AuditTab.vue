<template>
  <div class="audit-tab">
    <h3>🔗 勾稽规则</h3>

    <!-- 数值关系 -->
    <div v-if="arithmeticRules.length > 0" class="rule-section">
      <h4>📐 数值关系（{{ arithmeticRules.length }} 条）</h4>
      <el-table :data="arithmeticRules" border size="small">
        <el-table-column prop="rule" label="规则" />
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            {{ (row.confidence * 100).toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column label="优先级" width="80">
          <template #default="{ row }">
            <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
              {{ row.priority }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="violation_count" label="违反数" width="80" />
      </el-table>
    </div>

    <!-- 函数依赖 -->
    <div v-if="functionalRules.length > 0" class="rule-section">
      <h4>🏷️ 函数依赖（{{ functionalRules.length }} 条）</h4>
      <el-table :data="functionalRules" border size="small">
        <el-table-column prop="rule" label="规则" />
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            {{ (row.confidence * 100).toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column label="优先级" width="80">
          <template #default="{ row }">
            <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
              {{ row.priority }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 时序约束 -->
    <div v-if="temporalRules.length > 0" class="rule-section">
      <h4>📅 时序约束（{{ temporalRules.length }} 条）</h4>
      <el-table :data="temporalRules" border size="small">
        <el-table-column prop="rule" label="规则" />
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            {{ (row.confidence * 100).toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column label="优先级" width="80">
          <template #default="{ row }">
            <el-tag :type="row.priority === '高' ? 'danger' : row.priority === '中' ? 'warning' : 'info'" size="small">
              {{ row.priority }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <div v-if="totalRules === 0" class="empty-tip">
      未发现勾稽规则
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  reportData: { type: Object, default: () => ({}) },
  qualityData: { type: Object, default: () => ({}) }
})

const auditRules = computed(() => {
  return props.reportData?.quality_report?.audit_rules || {}
})

const arithmeticRules = computed(() => {
  return auditRules.value.arithmetic_rules || []
})

const functionalRules = computed(() => {
  return auditRules.value.functional_dependencies || []
})

const temporalRules = computed(() => {
  return auditRules.value.temporal_rules || []
})

const totalRules = computed(() => {
  return arithmeticRules.value.length + functionalRules.value.length + temporalRules.value.length
})
</script>

<style scoped>
.audit-tab h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.rule-section {
  margin-bottom: 24px;
}
.rule-section h4 {
  margin-bottom: 12px;
  color: #555;
  font-size: 14px;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #909399;
}
</style>