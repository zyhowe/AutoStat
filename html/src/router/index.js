import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Upload from '../views/Upload.vue'
import Quality from '../views/Quality.vue'
import DataOverview from '../views/DataOverview.vue'
import DataValidation from '../views/DataValidation.vue'
import PatternDiscovery from '../views/PatternDiscovery.vue'
import ConclusionSolution from '../views/ConclusionSolution.vue'
import ModelCenter from '../views/ModelCenter.vue'
import AIAssistant from '../views/AIAssistant.vue'
import Settings from '../views/Settings.vue'
// 暂时保留旧报告路由，后续删除
// import Report from '../views/Report.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/upload', name: 'Upload', component: Upload },
  { path: '/quality', name: 'Quality', component: Quality },
  { path: '/data-overview', name: 'DataOverview', component: DataOverview },
  { path: '/data-validation', name: 'DataValidation', component: DataValidation },
  { path: '/pattern-discovery', name: 'PatternDiscovery', component: PatternDiscovery },
  { path: '/conclusion-solution', name: 'ConclusionSolution', component: ConclusionSolution },
  { path: '/models', name: 'ModelCenter', component: ModelCenter },
  { path: '/ai', name: 'AIAssistant', component: AIAssistant },
  { path: '/settings', name: 'Settings', component: Settings },
  // 保留旧的 report 路由用于过渡，后续删除
  // { path: '/report', name: 'Report', component: Report },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router