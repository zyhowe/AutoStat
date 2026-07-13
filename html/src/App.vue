<!-- src/App.vue -->
<template>
  <div id="app">
    <el-container>
      <el-header height="60px">
        <Header />
      </el-header>
      <el-container>
        <el-aside width="220px">
          <Sidebar />
        </el-aside>
        <el-main>
          <router-view />
        </el-main>
      </el-container>
    </el-container>
    <!-- 全局字段详情弹窗 -->
    <FieldDetailDialog
      v-model="fieldDetailStore.visible"
      :field-name="fieldDetailStore.fieldName"
      :field-data="fieldDetailStore.fieldData"
    />
    <!-- 全局数据预览弹窗 -->
    <DataPreviewDialog
      v-model="previewStore.visible"
      :session-id="previewStore.sessionId"
      :title="previewStore.title"
      :filters="previewStore.filters"
      :fields="previewStore.fields"
      @close="previewStore.close"
    />
  </div>
</template>

<script setup>
import Header from './components/Header.vue'
import Sidebar from './components/Sidebar.vue'
import FieldDetailDialog from './components/FieldDetailDialog.vue'
import DataPreviewDialog from './components/DataPreviewDialog.vue'
import { useFieldDetailStore } from './stores/fieldDetail'
import { usePreviewStore } from './stores/preview'

const fieldDetailStore = useFieldDetailStore()
const previewStore = usePreviewStore()
</script>

<style>
#app {
  height: 100vh;
  overflow: hidden;
}
.el-container {
  height: 100vh;
}
.el-header {
  padding: 0;
  border-bottom: 1px solid #e4e7ed;
}
.el-aside {
  background-color: #f5f7fa;
  border-right: 1px solid #e4e7ed;
}
.el-main {
  padding: 20px;
  background-color: #ffffff;
  overflow-y: auto;
}
</style>