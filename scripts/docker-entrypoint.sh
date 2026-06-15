#!/bin/sh
set -e

# 解析 NEO4J_URI 中的主机和端口（默认 bolt://neo4j:7687）
NEO4J_HOST=${NEO4J_HOST:-neo4j}
NEO4J_PORT=${NEO4J_PORT:-7687}

echo "Waiting for Neo4j at ${NEO4J_HOST}:${NEO4J_PORT}..."
while ! nc -z "${NEO4J_HOST}" "${NEO4J_PORT}"; do
  sleep 1
done
echo "Neo4j is ready."

# 初始化 Neo4j schema
echo "Initializing Neo4j schema..."
python3 -m src.cli init

# 导入数据
echo "Importing data..."
python3 -m src.cli import --dir data

# 启动 FastAPI 服务
echo "Starting FastAPI server..."
exec python3 -m src.server
