<template>
  <div class="quality-score">
    <div class="score-overview">
      <div class="score-number">
        <span class="score-value">{{ data.overall_score || 0 }}</span>
        <span class="score-grade">{{ data.grade || '未知' }}</span>
      </div>
      <div class="score-label">综合评分</div>
    </div>

    <div class="score-dimensions">
      <div
        v-for="(score, name) in data.dimensions"
        :key="name"
        class="dimension-item"
      >
        <span class="dimension-name">{{ getLabel(name) }}</span>
        <el-progress
          :percentage="Math.round(score)"
          :color="getColor(score)"
          :stroke-width="8"
        />
        <span class="dimension-value">{{ Math.round(score) }}%</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

const props = defineProps({
  data: {
    type: Object,
    required: true,
    default: () => ({
      overall_score: 0,
      grade: '未知',
      dimensions: {}
    })
  }
})

const labelMap = {
  'completeness': '完整性',
  'accuracy': '准确性',
  'consistency': '一致性',
  'timeliness': '及时性',
  'uniqueness': '唯一性'
}

function getLabel(name) {
  return labelMap[name] || name
}

function getColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}
</script>

<style scoped>
.quality-score {
  display: flex;
  gap: 30px;
  align-items: center;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 8px;
}
.score-overview {
  text-align: center;
  min-width: 120px;
  padding-right: 30px;
  border-right: 1px solid #e4e7ed;
}
.score-number {
  display: flex;
  align-items: baseline;
  gap: 8px;
  justify-content: center;
}
.score-value {
  font-size: 36px;
  font-weight: bold;
  color: #2c3e50;
}
.score-grade {
  font-size: 16px;
  color: #909399;
  background: #fff;
  padding: 2px 12px;
  border-radius: 4px;
}
.score-label {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
.score-dimensions {
  flex: 1;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}
.dimension-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dimension-name {
  font-size: 12px;
  color: #666;
}
.dimension-value {
  font-size: 12px;
  font-weight: 500;
  text-align: right;
}
</style>