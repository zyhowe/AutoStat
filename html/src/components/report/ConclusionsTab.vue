<template>
  <div class="conclusions-tab">
    <h3>🎯 核心结论</h3>
    <div class="conclusion-cards">
      <el-card
        v-for="(conclusion, index) in conclusions"
        :key="index"
        class="conclusion-card"
        shadow="hover"
      >
        <div class="conclusion-icon">{{ conclusion.icon || '📌' }}</div>
        <div class="conclusion-title">{{ conclusion.title }}</div>
        <div class="conclusion-desc">{{ conclusion.description }}</div>
      </el-card>
    </div>
    <div v-if="!conclusions || conclusions.length === 0" class="empty-tip">
      暂无核心结论
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  reportData: { type: Object, default: () => ({}) },
  qualityData: { type: Object, default: () => ({}) }
})

const conclusions = computed(() => {
  // 从 reportData 中提取结论，或者从 summary 中获取
  return props.reportData?.summary || []
})
</script>

<style scoped>
.conclusions-tab h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.conclusion-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}
.conclusion-card {
  text-align: center;
  padding: 16px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.conclusion-icon {
  font-size: 28px;
  margin-bottom: 8px;
}
.conclusion-title {
  font-weight: bold;
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 4px;
}
.conclusion-desc {
  font-size: 12px;
  color: #909399;
  line-height: 1.4;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #909399;
}
</style>