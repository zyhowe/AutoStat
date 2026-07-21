<template>
  <div class="data-decode">
    <div v-if="!hasResults || totalRecords === 0" class="empty-records">
      <el-empty description="暂无明细记录，请先在「开始分析」执行场景">
        <el-button type="primary" @click="$emit('go-to-config')">去执行场景</el-button>
      </el-empty>
    </div>

    <div v-else class="records-content">
      <!-- 筛选栏 -->
      <div class="filter-bar">
        <el-select
          :model-value="filter.scenario"
          placeholder="按场景筛选"
          clearable
          size="small"
          style="width:180px;"
          @update:model-value="$emit('update:filter', { ...filter, scenario: $event })"
          @change="$emit('apply-filters')"
        >
          <el-option label="全部" value="" />
          <el-option
            v-for="s in scenarioResults"
            :key="s.scenario_id"
            :label="s.business_name || s.name"
            :value="String(s.scenario_id)"
          />
        </el-select>

        <el-input
          :model-value="filter.keyword"
          placeholder="搜索字段名/规则"
          size="small"
          style="width:180px;"
          clearable
          @update:model-value="$emit('update:filter', { ...filter, keyword: $event })"
          @input="$emit('apply-filters')"
        />

        <el-select
          :model-value="filter.recordType"
          placeholder="记录类型"
          clearable
          size="small"
          style="width:120px;"
          @update:model-value="$emit('update:filter', { ...filter, recordType: $event })"
          @change="$emit('apply-filters')"
        >
          <el-option label="全部" value="" />
          <el-option label="聚类" value="cluster" />
          <el-option label="勾稽" value="violation" />
          <el-option label="异常" value="outlier" />
          <el-option label="缺失" value="missing" />
          <el-option label="重复" value="duplicate" />
          <el-option label="集中" value="entity_concentration" />
        </el-select>

        <el-select
          :model-value="filter.severity"
          placeholder="严重程度"
          clearable
          size="small"
          style="width:120px;"
          @update:model-value="$emit('update:filter', { ...filter, severity: $event })"
          @change="$emit('apply-filters')"
        >
          <el-option label="全部" value="" />
          <el-option label="高" value="high" />
          <el-option label="中" value="medium" />
          <el-option label="低" value="low" />
        </el-select>

        <el-select
          :model-value="filter.status"
          placeholder="状态"
          clearable
          size="small"
          style="width:120px;"
          @update:model-value="$emit('update:filter', { ...filter, status: $event })"
          @change="$emit('apply-filters')"
        >
          <el-option label="全部" value="" />
          <el-option label="待核查" value="pending" />
          <el-option label="已忽略" value="ignored" />
          <el-option label="已处理" value="resolved" />
        </el-select>

        <el-button size="small" @click="$emit('reset-filters')">重置</el-button>
        <span class="filter-count">共 {{ filteredRecords.length }} 条记录</span>
        <el-button size="small" type="primary" plain @click="$emit('export')">📥 导出 CSV</el-button>
      </div>

      <!-- 记录表格 -->
      <div class="records-table">
        <el-table
          :data="pagedRecords"
          border
          size="small"
          max-height="500"
          style="width:100%;"
          v-loading="loading"
        >
          <el-table-column type="index" label="#" width="50" align="center" />
          <el-table-column prop="row" label="行号" width="80" align="center" sortable />
          <el-table-column prop="scenario_name" label="所属场景" min-width="120" sortable />
          <el-table-column prop="record_type_display" label="记录类型" width="100" align="center" sortable>
            <template #default="{ row }">
              <el-tag v-if="row.record_type === 'cluster'" size="small" type="primary">聚类</el-tag>
              <el-tag v-else-if="row.record_type === 'violation'" size="small" type="danger">勾稽</el-tag>
              <el-tag v-else-if="row.record_type === 'outlier'" size="small" type="warning">异常</el-tag>
              <el-tag v-else-if="row.record_type === 'missing'" size="small" type="info">缺失</el-tag>
              <el-tag v-else-if="row.record_type === 'duplicate'" size="small" type="warning">重复</el-tag>
              <el-tag v-else-if="row.record_type === 'entity_concentration'" size="small" type="success">集中</el-tag>
              <el-tag v-else size="small" type="info">其他</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="field_display" label="规则/字段" min-width="150" sortable>
            <template #default="{ row }">
              <span v-if="row.record_type === 'cluster'">归属群组</span>
              <span v-else-if="row.record_type === 'violation'">{{ row.rule || row.field_display || '—' }}</span>
              <span v-else>{{ row.field_display || row.field || '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="value_display" label="当前值" min-width="150" sortable>
            <template #default="{ row }">
              <span v-if="row.record_type === 'cluster'">群组 {{ row.cluster_id }}</span>
              <span v-else-if="row.record_type === 'entity_concentration'">{{ row.entity || '—' }}</span>
              <span v-else-if="row.record_type === 'violation' && row.values_display">
                {{ row.values_display }}
              </span>
              <span v-else-if="row.record_type === 'missing' && row.missing_fields">
                {{ row.missing_fields.join(', ') }}
              </span>
              <span v-else>{{ row.value !== undefined && row.value !== null ? row.value : '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="expected" label="预期值/范围" min-width="100">
            <template #default="{ row }">
              <span v-if="row.record_type === 'cluster' || row.record_type === 'entity_concentration' || row.record_type === 'duplicate'">—</span>
              <span v-else-if="row.record_type === 'violation'">相等</span>
              <span v-else>{{ row.expected || '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="deviation" label="偏离程度" width="100" align="center" sortable>
            <template #default="{ row }">
              <span v-if="row.record_type === 'cluster' || row.record_type === 'entity_concentration' || row.record_type === 'duplicate'">—</span>
              <span v-else>{{ row.deviation !== undefined ? row.deviation.toFixed(2) + 'x' : '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="severity" label="严重程度" width="90" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.severity" :type="row.severity === 'high' ? 'danger' : row.severity === 'medium' ? 'warning' : 'info'" size="small">{{ row.severity }}</el-tag>
              <span v-else>—</span>
            </template>
          </el-table-column>
          <el-table-column prop="status" label="状态" width="110" align="center">
            <template #default="{ row }">
              <el-select :model-value="row.status" size="small" placeholder="状态" @update:model-value="$emit('update-record-status', row, $event)">
                <el-option label="待核查" value="pending" />
                <el-option label="已忽略" value="ignored" />
                <el-option label="已处理" value="resolved" />
              </el-select>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80" align="center">
            <template #default="{ row }">
              <el-button size="small" text type="primary" @click="$emit('show-detail', row)">详情</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <!-- 分页 -->
      <div class="pagination">
        <el-pagination
          :model-value="currentPage"
          :page-size="pageSize"
          :total="filteredRecords.length"
          :page-sizes="[20, 50, 100, 200]"
          layout="total, sizes, prev, pager, next"
          @update:model-value="$emit('update:current-page', $event)"
          @update:page-size="$emit('update:page-size', $event)"
          @current-change="$emit('page-change')"
          @size-change="$emit('page-change')"
          size="small"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  hasResults: { type: Boolean, default: false },
  totalRecords: { type: Number, default: 0 },
  scenarioResults: { type: Array, default: () => [] },
  allRecords: { type: Array, default: () => [] },
  fieldMapping: { type: Object, default: () => ({}) },
  filter: { type: Object, default: () => ({ scenario: '', keyword: '', recordType: '', severity: '', status: '' }) },
  filteredRecords: { type: Array, default: () => [] },
  pagedRecords: { type: Array, default: () => [] },
  currentPage: { type: Number, default: 1 },
  pageSize: { type: Number, default: 50 },
  loading: { type: Boolean, default: false }
})

defineEmits([
  'apply-filters',
  'reset-filters',
  'page-change',
  'update-record-status',
  'show-detail',
  'export',
  'go-to-config',
  'update:filter',
  'update:current-page',
  'update:page-size'
])
</script>

<style scoped>
.data-decode { padding: 10px 0; }
.empty-records { padding: 40px 0; }

.filter-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 16px;
  background: #f5f7fa;
  border-radius: 8px;
  margin-bottom: 16px;
}
.filter-count { font-size: 12px; color: #909399; margin-left: auto; }

.records-table { margin-bottom: 16px; overflow-x: auto; }
.pagination { display: flex; justify-content: flex-end; }

@media (max-width: 768px) {
  .filter-bar { flex-direction: column; align-items: stretch; }
  .filter-count { margin-left: 0; }
}
</style>