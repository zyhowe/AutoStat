<template>
  <div class="header">
    <div class="logo">
      <span class="logo-icon">📊</span>
      <span class="logo-text">AutoStat</span>
      <span class="logo-version">v0.2.0</span>
      <span class="ip-tag">{{ clientIp }}</span>
    </div>
    <div class="header-right">
      <el-tag v-if="sessionStore.hasSession" type="success" size="small">
        当前项目: {{ sessionStore.sessionName }}
      </el-tag>
      <el-button size="small" @click="goHome">首页</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSessionStore } from '../stores/session'

const router = useRouter()
const sessionStore = useSessionStore()
const clientIp = ref('')

onMounted(() => {
  clientIp.value = window.location.hostname || ''
})

function goHome() {
  router.push('/')
}
</script>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  height: 60px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
}
.logo {
  display: flex;
  align-items: center;
  gap: 8px;
}
.logo-icon {
  font-size: 24px;
}
.logo-text {
  font-size: 20px;
  font-weight: bold;
  color: #2c3e50;
}
.logo-version {
  font-size: 12px;
  color: #909399;
  background: #f0f2f6;
  padding: 2px 8px;
  border-radius: 4px;
}
.ip-tag {
  font-size: 12px;
  color: #409eff;
  background: #ecf5ff;
  padding: 2px 12px;
  border-radius: 4px;
  border: 1px solid #d9ecff;
  margin-left: 12px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
</style>