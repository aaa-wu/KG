FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（neo4j-driver 等可能需要编译工具）
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件并安装，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露 FastAPI 服务端口
EXPOSE 8000

# 默认启动后端服务（docker-compose 中会被覆盖为完整启动流程）
CMD ["python3", "-m", "src.server"]
