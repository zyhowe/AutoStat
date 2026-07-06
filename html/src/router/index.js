import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Upload from '../views/Upload.vue'
import ReportSummary from '../views/ReportSummary.vue'
import Quality from '../views/Quality.vue'
import DataOverview from '../views/DataOverview.vue'
import DataValidation from '../views/DataValidation.vue'
import PatternDiscovery from '../views/PatternDiscovery.vue'
import ModelCenter from '../views/ModelCenter.vue'
import AIAssistant from '../views/AIAssistant.vue'
import Settings from '../views/Settings.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/upload', name: 'Upload', component: Upload },
  { path: '/report-summary', name: 'ReportSummary', component: ReportSummary },
  { path: '/quality', name: 'Quality', component: Quality },
  { path: '/data-overview', name: 'DataOverview', component: DataOverview },
  { path: '/data-validation', name: 'DataValidation', component: DataValidation },
  { path: '/pattern-discovery', name: 'PatternDiscovery', component: PatternDiscovery },
  { path: '/models', name: 'ModelCenter', component: ModelCenter },
  { path: '/ai', name: 'AIAssistant', component: AIAssistant },
  { path: '/settings', name: 'Settings', component: Settings }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router