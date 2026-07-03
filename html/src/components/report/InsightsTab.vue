<template>
  <div class="insights-tab">
    <h3>💡 智能解读</h3>
    <el-card shadow="hover">
      <el-timeline>
        <el-timeline-item
          v-for="(finding, index) in findings"
          :key="index"
          :type="index === 0 ? 'primary' : 'info'"
        >
          {{ finding }}
        </el-timeline-item>
      </el-timeline>
    </el-card>
    <div v-if="!findings || findings.length === 0" class="empty-tip">
      暂无智能解读
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  reportData: { type: Object, default: () => ({}) },
  qualityData: { type: Object, default: () => ({}) }
})

const findings = computed(() => {
  return props.reportData?.insights?.findings || []
})
</script>

<style scoped>
.insights-tab h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #909399;
}
</style>