import api from './index'

export const dataApi = {
  // 上传文件
  upload(file, sessionId = null) {
    const formData = new FormData()
    formData.append('file', file)
    if (sessionId) {
      formData.append('session_id', sessionId)
    }
    return api.post('/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  // 预览数据
  preview(file) {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/data/preview', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  // 加载示例数据
  loadDemo(dataset) {
    return api.post('/data/demo', null, { params: { dataset } })
  },

  // 加载数据库表
  loadDatabase(data) {
    return api.post('/data/database/load', data)
  }
}