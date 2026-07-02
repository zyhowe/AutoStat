<template>
  <div class="chart-container">
    <div ref="chartRef" :style="{ height: height, width: '100%' }"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, defineProps } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  options: {
    type: Object,
    required: true,
    default: () => ({})
  },
  height: {
    type: String,
    default: '300px'
  }
})

const chartRef = ref(null)
let chartInstance = null

onMounted(() => {
  initChart()
})

onBeforeUnmount(() => {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
})

watch(() => props.options, (newOptions) => {
  if (chartInstance && newOptions) {
    chartInstance.setOption(newOptions, true)
  }
}, { deep: true })

function initChart() {
  if (chartRef.value) {
    chartInstance = echarts.init(chartRef.value)
    chartInstance.setOption(props.options)

    // 响应式
    window.addEventListener('resize', handleResize)
  }
}

function handleResize() {
  if (chartInstance) {
    chartInstance.resize()
  }
}
</script>

<style scoped>
.chart-container {
  width: 100%;
}
</style>