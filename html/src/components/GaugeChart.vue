<template>
  <div ref="chartRef" class="gauge-chart"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  value: {
    type: Number,
    default: 0
  },
  max: {
    type: Number,
    default: 100
  }
})

const chartRef = ref(null)
let chartInstance = null
let resizeHandler = null

const renderChart = () => {
  if (!chartInstance) return
  const score = Number(props.value) || 0
  const color = score >= 80 ? '#67C23A' : score >= 60 ? '#E6A23C' : '#F56C6C'

  chartInstance.setOption({
    series: [{
      type: 'gauge',
      center: ['50%', '55%'],
      radius: '85%',
      startAngle: 210,
      endAngle: -30,
      min: 0,
      max: props.max,
      splitNumber: 5,
      progress: {
        show: true,
        width: 14,
        roundCap: true,
        itemStyle: { color: color }
      },
      axisLine: {
        lineStyle: {
          width: 14,
          color: [
            [0.3, '#F56C6C'],
            [0.7, '#E6A23C'],
            [1, '#67C23A']
          ]
        }
      },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      pointer: { show: false },
      anchor: { show: false },
      title: { show: false },
      detail: {
        valueAnimation: true,
        formatter: function(params) {
          // ✅ 兼容不同的参数格式
          const val = typeof params === 'object' ? (params.value || 0) : (params || 0)
          return Number(val).toFixed(1) + ' 分'
        },
        color: '#2C3E50',
        fontSize: 24,
        fontWeight: 'bold',
        offsetCenter: [0, '30%']
      },
      data: [{ value: score }]
    }]
  }, true)
}

onMounted(() => {
  nextTick(() => {
    if (chartRef.value) {
      chartInstance = echarts.init(chartRef.value)
      renderChart()
      resizeHandler = () => chartInstance?.resize()
      window.addEventListener('resize', resizeHandler)
    }
  })
})

onBeforeUnmount(() => {
  if (resizeHandler) {
    window.removeEventListener('resize', resizeHandler)
    resizeHandler = null
  }
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
})

watch(() => props.value, () => {
  renderChart()
}, { immediate: false })
</script>

<style scoped>
.gauge-chart {
  width: 100%;
  height: 180px;
}
</style>