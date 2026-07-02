import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { sessionApi } from '../api/session'
import { dataApi } from '../api/data'

export const useSessionStore = defineStore('session', () => {
  // State
  const currentSessionId = ref(null)
  const projects = ref([])
  const currentSession = ref(null)
  const uploadData = ref(null)

  // Getters
  const hasSession = computed(() => !!currentSessionId.value)
  const sessionName = computed(() => currentSession.value?.source_name || '未命名')

  // Actions
  async function createSession(sourceName = '未命名') {
    const result = await sessionApi.create({
      source_name: sourceName,
      analysis_type: 'single'
    })
    currentSessionId.value = result.session_id
    await loadSession(result.session_id)
    return result
  }

  async function loadSession(sessionId) {
    const session = await sessionApi.get(sessionId)
    currentSession.value = session
    currentSessionId.value = sessionId
    return session
  }

  async function loadProjects() {
    const result = await sessionApi.list()
    projects.value = result.projects
    return projects.value
  }

  async function deleteSession(sessionId) {
    await sessionApi.delete(sessionId)
    await loadProjects()
    if (currentSessionId.value === sessionId) {
      currentSessionId.value = null
      currentSession.value = null
    }
  }

  async function uploadFile(file) {
    const result = await dataApi.upload(file, currentSessionId.value)
    uploadData.value = result
    return result
  }

  function clearUpload() {
    uploadData.value = null
  }

  return {
    // State
    currentSessionId,
    projects,
    currentSession,
    uploadData,
    // Getters
    hasSession,
    sessionName,
    // Actions
    createSession,
    loadSession,
    loadProjects,
    deleteSession,
    uploadFile,
    clearUpload
  }
})