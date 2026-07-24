// src/api/index.js
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    return Promise.reject({ message, status: error.response?.status })
  }
)

// ==================== 导出流式查询 API（新增） ====================
// stream.js 是新增的独立文件
export { streamApi, streamQuery, streamPreview } from './stream'

// ==================== 默认导出（保持原有） ====================
export default api