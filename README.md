# KG 知识图谱学习路径推荐

这是一个基于 Neo4j 的课程知识图谱原型，包含：

- 专业、课程、知识点和前置依赖关系建模
- `entities.csv` / `relations.csv` 真实数据导入
- 知识点前驱链与学习路径推荐
- 跨学校同名专业课程/知识点覆盖对比
- FastAPI 后端和 3D 图谱前端可视化
- DeepSeek/OpenAI 兼容接口支持的交互式学习规划

## 环境要求

- Python 3.9+
- Neo4j 5.x
- 可选：DeepSeek API Key，用于 `chat` 交互模式

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 配置

在项目根目录创建 `.env`：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

`.env` 包含本地密钥和数据库密码，不应提交到 Git；当前 `.gitignore` 已忽略它。

## 初始化和导入数据

确保 Neo4j 已启动后，先初始化约束和索引：

```bash
python3 -m src.cli init
```

导入当前真实数据：

```bash
python3 -m src.cli import --dir data
```

导入命令会清空当前 Neo4j 数据库中的所有节点和关系，再写入指定目录数据。

## 启动后端和前端

启动 FastAPI 服务：

```bash
python3 -m src.server
```

默认监听 `http://localhost:8000`。启动后在浏览器打开：

```text
http://localhost:8000
```

当前 `data` 使用实体-关系 CSV 格式。新数据可以只提供实体文件，导入器会先构建节点；如果存在关系文件，会一并构建关系：

```text
entities_final.csv           id,label,name  （优先使用）
entities.csv                 id,label,name  （备选）
relations.csv                start_id,type,end_id  （可选）
knowledge_concept_audit.csv  关系来源和置信度审计信息
```

常用接口：

```text
GET /api/stats
GET /api/graph?limit=5000
GET /api/search?q=深度学习
GET /api/graph/prerequisites/深度学习
```

## CLI 示例

查看帮助：

```bash
python3 -m src.cli --help
```

查找知识点完整前驱学习路径：

```bash
python3 -m src.cli path --target 深度学习
```

指定已掌握知识点后，返回仍需学习的完整依赖差集：

```bash
python3 -m src.cli path --target 深度学习 --known 矩阵运算,特征值与特征向量
```

对比两校同名专业：

```bash
python3 -m src.cli compare --major 计算机科学与技术 --uni-a 清华大学 --uni-b 北京大学
```

进入自然语言交互模式：

```bash
python3 -m src.cli chat
```

## 当前范围

`path --target` 当前面向知识点名称。课程或专业级路径规划可先通过 `chat` 模式探索，后续可以扩展成显式 CLI 子命令。
