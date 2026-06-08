# Docker 部署指南

把 milu 内置 Web 服务（`milu serve`：AgentPool 多用户对话 + SSE 流式 + 演示前端 + 嵌入式定时任务调度）打包为容器一键部署。

## 快速开始（docker compose，推荐）

```bash
# 1. 准备密钥：复制模板并填入至少一个厂商的 API Key
cp .env.example .env

# 2. 构建并启动（国内网络见下方「构建加速」）
docker compose up -d

# 3. 访问 http://localhost:8000 ；查看日志
docker compose logs -f milu
```

或者不用 compose，裸 docker：

```bash
docker build -t milu .
docker run -d --name milu -p 8000:8000 --env-file .env -v milu-data:/data/milu milu
```

## 构建加速（国内网络）

PyPI 源经 `PIP_INDEX_URL` 构建参数切换：

```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t milu .
```

compose 用户取消 `docker-compose.yml` 中 `args: PIP_INDEX_URL` 的注释即可。

## 镜像说明

| 设计 | 说明 |
|------|------|
| 多阶段构建 | builder 阶段装依赖进 venv，运行镜像只携带 venv（python:3.12-slim 基底） |
| 依赖范围 | 完整 milu：核心已含 Web 服务、向量知识库与 MCP 协议（无需额外 extra） |
| 非 root 运行 | 进程以 `milu`（uid 1000）用户运行 |
| 健康检查 | 内置 `HEALTHCHECK` 探测 `GET /health`（30s 间隔） |
| 数据持久化 | `MILU_HOME=/data/milu` 挂数据卷：会话、记忆、知识库、定时任务、用户级配置都在这里 |
| 时区 | 容器默认 UTC；定时任务按本地时间执行，国内部署建议 `TZ=Asia/Shanghai`（compose 已默认） |

## 配置注入

遵循 milu 的「密钥进 .env、参数进 config.json」分层（详见 README）：

1. **API Key**：`--env-file .env` 或 `-e DEEPSEEK_API_KEY=sk-xxx` 注入进程环境变量（优先级最高，不需要把 .env 拷进镜像）。
2. **可调参数**（默认厂商/模型/模式、运行限额、知识库开关等）：把项目的 `config/` 目录挂载到 `/app/config`（`MILU_PROJECT_DIR=/app` 已内置），容器内即按 `config/milu.json` 解析。
3. **启动参数**：覆盖容器命令即可，例如指定厂商与端口：
   ```bash
   docker run ... milu milu serve --host 0.0.0.0 --port 8000 -p deepseek --mode manual
   ```
   ⚠️ `--host 0.0.0.0` 不可省略——容器内绑默认的 127.0.0.1 时宿主机访问不到。

## MCP 注意事项

- `streamable_http` / `sse` 型 MCP 服务器：直接在 `config/mcp_servers.json` 配 URL 即可用。
- `stdio` 型 MCP 服务器：需要容器内有对应运行时（如 `npx` 需 Node.js）。基础镜像**未内置 Node**，需要时自行扩展：
  ```dockerfile
  FROM milu:latest
  USER root
  RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
  USER milu
  ```
- 多用户场景建议共享 MCP（整池一组 MCP 子进程），见 README「部署建议」。

## 升级与数据

```bash
docker compose build --pull   # 重建镜像
docker compose up -d          # 滚动替换容器
```

数据卷 `milu-data` 独立于容器生命周期，升级/重建不丢会话与知识库。备份：

```bash
docker run --rm -v milu-data:/data -v ${PWD}:/backup alpine tar czf /backup/milu-data.tar.gz -C /data .
```

## 生产建议

- **HTTPS/域名**：前置 Nginx/Caddy 反向代理；SSE 需关闭代理缓冲（Nginx：`proxy_buffering off;`）。
- **多副本**：单容器即单进程 AgentPool。横向扩容时按 `user_id` 做粘性路由（如 Nginx `ip_hash`），同会话由进程内 entry 锁串行，无需分布式锁——详见 README「部署建议（多 worker / 高并发）」。
- **定时任务单实例**：调度引擎有单实例锁（`MILU_HOME` 内 PID 文件），多副本共挂同一数据卷时只有一个副本执行定时任务，其余自动等待接管；也可只给一个副本开调度（其余加 `--no-scheduler`）。
- **资源限额**：`AgentPool` 的 `max_agents` / `max_concurrent_runs` 等经挂载的 `config/milu.json` `pool` 分节调整。
