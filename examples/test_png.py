import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建有向图
G = nx.DiGraph()

# 定义各层节点和层级编号（数字越小越靠上）
layers = {
    '前端展示层': {
        'nodes': ['动态表单/列表', 'AI对话编辑页', '规则编辑器', '溯源高亮面板', 'PDF区域框选',
                  '批量操作面板', '任务管理界面', '分析报告视图', '操作回放面板', '数据质量仪表板',
                  '数据血缘图谱', '主数据管理界面', '消息中心', '权限管理界面', '任务分配看板',
                  '数据集管理界面', '生命周期策略配置', '数据订阅配置', 'SLA监控配置', '审批流界面'],
        'level': 0
    },
    '后端核心服务层': {
        'nodes': ['界面配置服务', '校验引擎', '数据服务', '规则挖掘调度', '智能采集服务',
                  '批量操作服务', '自动任务调度器', '数据分析服务', '行为学习服务', '异常监控服务',
                  '数据版本服务', '数据质量服务', '数据血缘服务', '主数据服务', '数据安全服务',
                  '数据集成服务', '数据运营服务', '数据测试服务', '统一日志服务', '权限管理服务',
                  '消息服务', '数据集服务', '审批流服务', '生命周期策略服务', '订阅推送服务', 'SLA监控服务'],
        'level': 1
    },
    '中间件与持久化层': {
        'nodes': ['SQL Server\\n主从/分片', 'Redis\\n集群', 'Elasticsearch\\n集群', 'MongoDB\\n热日志',
                  'MinIO/OSS\\n对象存储', '消息队列\\nKafka/RabbitMQ', 'CDN\\n冷日志归档'],
        'level': 2
    },
    '扩展服务与引擎': {
        'nodes': ['规则挖掘算法服务', '大模型API', 'WPS文档服务', 'OCR引擎', '基础HTTP抓取器',
                  'JS渲染抓取器', 'Playwright抓取器', '语音识别服务', '日志采集与流处理',
                  '数据质量引擎', '血缘采集器', '加密/脱敏服务', '数据同步引擎', '消息推送网关'],
        'level': 3
    }
}

# 添加节点，并存储层级信息
for layer_name, layer_info in layers.items():
    for node in layer_info['nodes']:
        G.add_node(node, layer=layer_info['level'], layername=layer_name)

# 定义主要的边（前端->后端）
front_to_back = [
    ('动态表单/列表', '界面配置服务'), ('AI对话编辑页', '智能采集服务'), ('规则编辑器', '规则挖掘调度'),
    ('溯源高亮面板', '智能采集服务'), ('PDF区域框选', '智能采集服务'), ('批量操作面板', '批量操作服务'),
    ('任务管理界面', '自动任务调度器'), ('分析报告视图', '数据分析服务'), ('操作回放面板', '行为学习服务'),
    ('数据质量仪表板', '数据质量服务'), ('数据血缘图谱', '数据血缘服务'), ('主数据管理界面', '主数据服务'),
    ('消息中心', '消息服务'), ('权限管理界面', '权限管理服务'), ('任务分配看板', '权限管理服务'),
    ('数据集管理界面', '数据集服务'), ('生命周期策略配置', '生命周期策略服务'), ('数据订阅配置', '订阅推送服务'),
    ('SLA监控配置', 'SLA监控服务'), ('审批流界面', '审批流服务')
]
for src, tgt in front_to_back:
    G.add_edge(src, tgt, color='blue', style='solid')

# 后端->中间件
back_to_mw = [
    ('数据服务', 'SQL Server\\n主从/分片'), ('数据服务', 'Redis\\n集群'), ('数据服务', 'Elasticsearch\\n集群'),
    ('规则挖掘调度', 'SQL Server\\n主从/分片'), ('批量操作服务', 'SQL Server\\n主从/分片'),
    ('自动任务调度器', 'SQL Server\\n主从/分片'), ('数据分析服务', 'SQL Server\\n主从/分片'),
    ('数据分析服务', 'Elasticsearch\\n集群'), ('行为学习服务', 'Redis\\n集群'),
    ('异常监控服务', '消息队列\\nKafka/RabbitMQ'), ('数据版本服务', 'SQL Server\\n主从/分片'),
    ('数据质量服务', 'SQL Server\\n主从/分片'), ('数据血缘服务', 'SQL Server\\n主从/分片'),
    ('主数据服务', 'SQL Server\\n主从/分片'), ('数据安全服务', 'SQL Server\\n主从/分片'),
    ('数据集成服务', 'SQL Server\\n主从/分片'), ('数据运营服务', 'SQL Server\\n主从/分片'),
    ('数据测试服务', 'SQL Server\\n主从/分片'), ('统一日志服务', 'MongoDB\\n热日志'),
    ('统一日志服务', 'MinIO/OSS\\n对象存储'), ('统一日志服务', 'CDN\\n冷日志归档'),
    ('权限管理服务', 'SQL Server\\n主从/分片'), ('消息服务', 'SQL Server\\n主从/分片'),
    ('数据集服务', 'MinIO/OSS\\n对象存储'), ('审批流服务', 'SQL Server\\n主从/分片'),
    ('生命周期策略服务', 'SQL Server\\n主从/分片'), ('订阅推送服务', '消息队列\\nKafka/RabbitMQ'),
    ('SLA监控服务', 'SQL Server\\n主从/分片')
]
for src, tgt in back_to_mw:
    G.add_edge(src, tgt, color='orange', style='solid')

# 扩展服务连接
ext_links = [
    ('规则挖掘调度', '规则挖掘算法服务'), ('智能采集服务', 'OCR引擎'), ('智能采集服务', '基础HTTP抓取器'),
    ('智能采集服务', 'JS渲染抓取器'), ('智能采集服务', 'Playwright抓取器'), ('AI对话编辑页', '大模型API'),
    ('AI对话编辑页', '语音识别服务'), ('异常监控服务', '日志采集与流处理'), ('数据质量服务', '数据质量引擎'),
    ('数据血缘服务', '血缘采集器'), ('数据安全服务', '加密/脱敏服务'), ('数据集成服务', '数据同步引擎'),
    ('消息服务', '消息推送网关')
]
for src, tgt in ext_links:
    G.add_edge(src, tgt, color='purple', style='dashed')

# 计算布局：按层级垂直排列
pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')

# 绘制
plt.figure(figsize=(24, 20))
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
layer_nodes = {0: [], 1: [], 2: [], 3: []}
for node, attr in G.nodes(data=True):
    layer = attr['layer']
    layer_nodes[layer].append(node)

# 按层绘制节点和颜色
for layer, nodes in layer_nodes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[layer], node_size=2000, node_shape='s')
    # 添加层级标签
    for node in nodes:
        x, y = pos[node]
        plt.text(x, y - 0.15, node, ha='center', va='top', fontsize=7, fontfamily='sans-serif')

# 绘制边（按颜色）
for edge in G.edges(data=True):
    src, tgt, attr = edge
    color = attr.get('color', 'gray')
    style = attr.get('style', 'solid')
    nx.draw_networkx_edges(G, pos, edgelist=[(src, tgt)], edge_color=color, style=style, width=1.5, arrowsize=10)

# 添加层级标题
for layer_name, layer_info in layers.items():
    level = layer_info['level']
    y_pos = 1 - level * 0.25  # 根据层级计算Y位置
    plt.text(0, y_pos, layer_name, transform=plt.gcf().transFigure, fontsize=12, fontweight='bold', va='center')

plt.title("智能数据管理平台逻辑架构图", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('platform_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("架构图已保存为 platform_architecture.png")