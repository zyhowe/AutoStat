// src/main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'
import router from './router'
import './styles/main.css'
import { useAnalysisStore } from './stores/analysis'

// ==================== ECharts 全局注册 ====================
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import {
  BarChart,
  PieChart,
  RadarChart,
  GaugeChart,
  ScatterChart,
  HeatmapChart,
  BoxplotChart,
  LineChart
} from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  VisualMapComponent,
  ToolboxComponent,
  DataZoomComponent,
  MarkLineComponent  // ✅ 新增：注册 MarkLine 组件
} from 'echarts/components'

use([
  CanvasRenderer,
  BarChart,
  PieChart,
  RadarChart,
  GaugeChart,
  ScatterChart,
  HeatmapChart,
  BoxplotChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  VisualMapComponent,
  ToolboxComponent,
  DataZoomComponent,
  MarkLineComponent  // ✅ 新增
])

const app = createApp(App)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.component('v-chart', VChart)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

app.mount('#app')

const analysisStore = useAnalysisStore()
analysisStore.setRouter(router)