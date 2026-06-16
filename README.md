# KG 知识图谱学习路径推荐

基于 Neo4j 的课程知识图谱原型，支持专业-课程-知识点建模、前置依赖分析、LLM 增强的 Topic 抽取与学习路径推荐。

## 功能特性

- **专业、课程、知识点和前置依赖关系建模**
- **Topic / SubTopic / Domain 本体层**，支持 LLM 自动抽取与人工审批
- **实体-关系 CSV 数据导入**（`entities_final.csv` / `relations_final.csv`）
- **知识点前驱链与学习路径推荐**
- **语义相似度计算**（sentence-transformers）
- **跨学校同名专业课程/知识点覆盖对比**
- **FastAPI 后端 + 3D 图谱前端可视化**
- **DeepSeek/OpenAI 兼容接口支持的交互式学习规划**

> ⚠️ **注意**：LLM 自动预测前置关系功能默认关闭。此前曾出现将“AI”与“生物神经科学”混淆的荒谬路径，建议仅在人工复核后启用。

## 目录结构

```text
.
├── data/                       # 数据文件（已包含在仓库中）
│   ├── entities_final.csv      # 实体数据
│   ├── relations_final.csv     # 关系数据
│   ├── knowledge_embeddings.pkl # 语义嵌入模型
│   └── ...
├── src/
│   ├── cli.py                  # 命令行入口
│   ├── server.py               # FastAPI 后端服务
│   ├── ingestion/              # 数据解析与导入
│   ├── models/                 # Neo4j Schema 定义
│   ├── recommendation/         # 路径推荐算法
│   ├── prereq_prediction/      # 前置关系预测
│   ├── ontology/               # Topic 抽取与审批
│   └── rl/                     # 双图 RL 推荐
├── static/                     # 前端页面
│   ├── index.html              # 3D 图谱可视化
│   └── admin.html              # 管理后台
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
└── README.md                   # 本文件
```

## 环境要求

- Docker + Docker Compose（**推荐，一键启动**）
- 或手动安装：Python 3.9+、Neo4j 5.x
- 有效的 DeepSeek API Key（用于 chat、Topic 抽取、前置关系补全等增强功能）

## 快速开始（推荐：Docker）

只需要安装 [Docker](https://www.docker.com/)，无需手动配置 Python 和 Neo4j。

### 1. 克隆仓库

```bash
git clone https://github.com/aaa-wu/KG.git
cd KG
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 `DEEPSEEK_API_KEY`：

```bash
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 🔒 `.env` 包含密钥和数据库密码，**不应提交到 Git**，`.gitignore` 已忽略它。

### 3. 启动全部服务

```bash
docker compose up --build
```

首次启动会自动：
1. 拉取并启动 Neo4j 容器
2. 等待 Neo4j 就绪
3. 初始化 Schema
4. 导入 `data/` 目录数据
5. 启动 FastAPI 服务

### 4. 打开前端

浏览器访问：

```text
http://localhost:8000
```

管理后台：

```text
http://localhost:8000/admin.html
```

Neo4j Browser（可选）：

```text
http://localhost:7474
```

默认账号：`neo4j` / 你在 `.env` 中设置的密码。

### 停止服务

```bash
# 前台运行时按 Ctrl+C
# 或后台停止
docker compose down

# 想清空数据库数据
docker compose down -v
```

---

## 手动安装（备选）

如果你不想用 Docker，可以按以下步骤手动安装。

### 1. 克隆仓库

```bash
git clone https://github.com/aaa-wu/KG.git
cd KG
```

### 2. 安装 Python 依赖

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 安装并启动 Neo4j

- 下载 Neo4j 5.x：[https://neo4j.com/download/](https://neo4j.com/download/)
- 启动后记住设置的密码
- 默认访问地址：`bolt://localhost:7687`

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

### 5. 初始化并导入数据

```bash
python3 -m src.cli init
python3 -m src.cli import --dir data
```

> 导入命令会**清空当前 Neo4j 数据库中的所有节点和关系**，再写入指定目录数据。

### 6. 启动后端服务

```bash
python3 -m src.server
```

默认监听 `http://localhost:8000`。

### 7. 打开前端

浏览器访问：

```text
http://localhost:8000
```

管理后台：

```text
http://localhost:8000/admin.html
```

## 验证安装

服务启动后，可以访问以下接口测试：

```text
GET http://localhost:8000/api/stats
GET http://localhost:8000/api/graph?limit=5000
GET http://localhost:8000/api/search?q=深度学习
GET http://localhost:8000/api/graph/prerequisites/深度学习
```

## CLI 常用命令

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

## LLM 辅助数据增强

### 1. Topic/SubTopic 自动抽取

从现有课程名和知识点名中，用 DeepSeek 推断知识模块层次：

```bash
python3 -m src.cli extract-topics --major 档案学 --queue
python3 -m src.cli validate list
python3 -m src.cli validate approve --id <待审项ID>
```

### 2. 前置关系 LLM 补全（默认关闭，使用前请确认）

```bash
# 只预览 10 个目标的效果
python3 -m src.cli predict-prereq --method llm --dry-run true --max-targets 10

# 确认效果后再正式写入（请谨慎）
python3 -m src.cli predict-prereq --method llm --dry-run false --max-targets 2520 --threshold 0.7

# 如果 DeepSeek 不可用，可回退到老算法
python3 -m src.cli predict-prereq --method mlp --dry-run false
```

### 3. 语义相似度计算

```bash
python3 -m src.cli similarity --label KnowledgeConcept --threshold 0.75
```

### 4. 双图 RL 推荐

```bash
python3 -m src.cli recommend --target 分治算法 --known 算法 --level beginner
```

## 数据文件说明

当前 `data` 目录使用实体-关系 CSV 格式：

```text
entities_final.csv           id,label,name  （优先使用）
entities.csv                 id,label,name  （备选）
relations.csv                start_id,type,end_id  （可选）
knowledge_concept_audit.csv  关系来源和置信度审计信息
```

新数据可以只提供实体文件，导入器会先构建节点；如果存在关系文件，会一并构建关系。

## API 端点

```text
GET  /api/stats
GET  /api/graph?limit=5000
GET  /api/search?q=深度学习
GET  /api/graph/prerequisites/深度学习

POST /api/extract/topics              抽取 Topic/SubTopic
POST /api/extract/topics/queue        抽取并加入复核队列
GET  /api/validation/pending          待审列表
POST /api/validation/approve          批准并导入
GET  /api/courses/{name}/topics       课程主题层级
POST /api/similarity/compute          计算语义相似度
GET  /api/similar/concepts/{name}     相似概念

POST /api/prereq/predict              前置关系补全
GET  /api/prereq/predicted            预测前置边列表
POST /api/recommend                   双图 RL 个性化推荐
```


