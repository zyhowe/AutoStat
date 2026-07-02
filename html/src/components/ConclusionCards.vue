<template>
  <div class="conclusion-cards">
    <el-card
      v-for="(item, index) in data"
      :key="index"
      class="conclusion-card"
      shadow="hover"
    >
      <div class="conclusion-icon">{{ item.icon || '📌' }}</div>
      <div class="conclusion-title">{{ item.title }}</div>
      <div class="conclusion-desc">{{ item.description }}</div>
    </el-card>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

defineProps({
  data: {
    type: Array,
    required: true,
    default: () => []
  }
})
</script>

<style scoped>
.conclusion-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
  font-size: 24px;
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
</style>