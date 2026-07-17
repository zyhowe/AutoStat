import api from './index'

export const dataApi = {
  // 单文件上传
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

  // 多文件上传
  uploadMulti(files, sessionId = null) {
    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }
    if (sessionId) {
      formData.append('session_id', sessionId)
    }
    return api.post('/data/upload/multi', formData, {
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
  },

  // 确认关系
  confirmRelations(data) {
    return api.post('/data/relations/confirm', data)
  },

  // 获取关系
  getRelations(sessionId) {
    return api.get(`/data/relations/${sessionId}`)
  },

  // ===== 新增：更新字段类型 =====
  updateFieldTypes(sessionId, tableName, fieldTypes) {
    return api.post('/data/field_types/update', {
      session_id: sessionId,
      table_name: tableName || 'merged',
      field_types: fieldTypes
    })
  }
}