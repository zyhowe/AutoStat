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
    <GlobalBubble />
  </div>
</template>

<script setup>
import Header from './components/Header.vue'
import Sidebar from './components/Sidebar.vue'
import GlobalBubble from './components/GlobalBubble.vue'
</script>

<style>
#app { height: 100vh; overflow: hidden; }
.el-container { height: 100vh; }
.el-header { padding: 0; border-bottom: 1px solid #e4e7ed; }
.el-aside { background-color: #f5f7fa; border-right: 1px solid #e4e7ed; }
.el-main { padding: 20px; background-color: #ffffff; overflow-y: auto; }
</style>