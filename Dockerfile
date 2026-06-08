# syntax=docker/dockerfile:1
# ════════════════════════════════════════════════════════════
# milu 内置 Web 服务（milu serve）容器镜像
#
# 构建：    docker build -t milu .
# 国内加速：docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t milu .
# 运行：    docker run -d -p 8000:8000 --env-file .env -v milu-data:/data/milu milu
# 推荐用 docker compose（见 docker-compose.yml 与 docs/Docker部署.md）
# ════════════════════════════════════════════════════════════

# ── 构建阶段：venv 中安装 milu 及 Web 服务依赖 ──
FROM python:3.12-slim AS builder

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /build
# hatchling 构建需要包元数据 + 源码（pyproject 声明了 readme 与 license 文件）
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -i "$PIP_INDEX_URL" --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -i "$PIP_INDEX_URL" "."

# ── 运行阶段：仅携带 venv 的精简镜像 ──
FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv

# MILU_HOME：会话/记忆/知识库/定时任务等「写数据」目录（挂数据卷持久化）
# MILU_PROJECT_DIR：项目级「读配置」目录（挂载 ./config 到 /app/config 即生效）
ENV PATH="/opt/venv/bin:$PATH" \
    MILU_HOME=/data/milu \
    MILU_PROJECT_DIR=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN useradd --create-home --uid 1000 milu \
    && mkdir -p /data/milu \
    && chown -R milu:milu /data/milu /app
USER milu

VOLUME /data/milu
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# 容器内必须绑 0.0.0.0 才能从宿主机访问；API Key 经 --env-file/-e 注入
CMD ["milu", "serve", "--host", "0.0.0.0", "--port", "8000"]
