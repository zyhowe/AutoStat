<template>
  <div class="cleaning-tab">
    <h3>🧹 清洗建议</h3>
    <el-timeline>
      <el-timeline-item
        v-for="(suggestion, index) in suggestions"
        :key="index"
        :type="index === 0 ? 'primary' : 'info'"
      >
        {{ suggestion }}
      </el-timeline-item>
    </el-timeline>
    <div v-if="!suggestions || suggestions.length === 0" class="empty-tip">
      ✅ 数据质量良好，无需清洗
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  reportData: { type: Object, default: () => ({}) },
  qualityData: { type: Object, default: () => ({}) }
})

const suggestions = computed(() => {
  return props.reportData?.cleaning_suggestions || []
})
</script>

<style scoped>
.cleaning-tab h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #67c23a;
  font-size: 16px;
}
</style>