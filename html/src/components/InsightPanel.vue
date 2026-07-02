<template>
  <div class="insight-panel">
    <el-timeline>
      <el-timeline-item
        v-for="(item, index) in data"
        :key="index"
        :type="index === 0 ? 'primary' : 'info'"
        :icon="item.icon"
      >
        <div class="insight-item">
          <div class="insight-title">{{ item.title }}</div>
          <div class="insight-desc">{{ item.description }}</div>
        </div>
      </el-timeline-item>
    </el-timeline>
    <div v-if="data.length === 0" class="empty-text">
      <el-empty description="暂无洞察" />
    </div>
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
.insight-panel {
  padding: 10px 0;
}
.insight-item {
  padding: 4px 0;
}
.insight-title {
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 2px;
}
.insight-desc {
  font-size: 13px;
  color: #666;
}
.empty-text {
  padding: 20px 0;
}
</style>