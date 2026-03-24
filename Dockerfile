# 使用官方 Python 基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml version ./
COPY src/ ./src/
COPY README.md LICENSE ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -e "."

# 安装 FastAPI、Uvicorn 和 python-multipart（用于 CSV 文件上传）
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# 复制应用代码
COPY app.py ./
COPY examples/ ./examples/

# 暴露端口
EXPOSE 8000

# 创建入口脚本，支持多种运行模式
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting REST API server..."\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "demo" ]; then\n\
    echo "Running demo script..."\n\
    exec python examples/demo.py\n\
elif [ "$1" = "train" ]; then\n\
    echo "Running training script..."\n\
    shift\n\
    exec python "$@"\n\
elif [ "$1" = "python" ]; then\n\
    shift\n\
    exec python "$@"\n\
else\n\
    echo "TCLF Trade Classification Tool"\n\
    echo ""\n\
    echo "Running demo script by default..."\n\
    echo "Use one of the following commands to change behavior:"\n\
    echo ""\n\
    echo "  docker run <image> api          - Start REST API server"\n\
    echo "  docker run <image> demo         - Run demo script"\n\
    echo "  docker run <image> train <file> - Run training script"\n\
    echo "  docker run <image> python <cmd> - Run Python command"\n\
    echo ""\n\
    exec python examples/demo.py\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]

# 默认命令（运行演示脚本）
CMD ["demo"]
