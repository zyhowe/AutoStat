<template>
  <div class="file-upload">
    <el-upload
      ref="uploadRef"
      drag
      :auto-upload="false"
      :on-change="handleFileChange"
      :on-remove="handleRemove"
      :limit="1"
      action="#"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        拖拽文件到此处，或 <em>点击选择</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">
          {{ tip }}
        </div>
      </template>
    </el-upload>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue'

const props = defineProps({
  tip: {
    type: String,
    default: '支持 CSV, Excel, JSON, TXT 格式'
  },
  accept: {
    type: String,
    default: '.csv,.xlsx,.xls,.json,.txt'
  }
})

const emit = defineEmits(['change', 'remove'])

const uploadRef = ref(null)

function handleFileChange(file) {
  emit('change', file.raw)
}

function handleRemove() {
  emit('remove')
}

function clearFiles() {
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}

defineExpose({
  clearFiles
})
</script>

<style scoped>
.file-upload {
  width: 100%;
}
</style>