import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Upload from '../views/Upload.vue'
import Quality from '../views/Quality.vue'
import Report from '../views/Report.vue'
import Compare from '../views/Compare.vue'
import ModelCenter from '../views/ModelCenter.vue'
import AIAssistant from '../views/AIAssistant.vue'
import Settings from '../views/Settings.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/upload', name: 'Upload', component: Upload },
  { path: '/quality', name: 'Quality', component: Quality },
  { path: '/report', name: 'Report', component: Report },
  { path: '/compare', name: 'Compare', component: Compare },
  { path: '/models', name: 'ModelCenter', component: ModelCenter },
  { path: '/ai', name: 'AIAssistant', component: AIAssistant },
  { path: '/settings', name: 'Settings', component: Settings }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router