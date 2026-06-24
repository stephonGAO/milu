# milu Gateway —— 多渠道接入（微信客服 / 飞书 / Telegram）

把 milu Agent 接到各 IM 平台。采用 **Ports & Adapters（六边形）** 架构：平台无关的
「接入核心」与每个平台的「适配器 Channel」解耦，新增平台 = 只写一个 Channel，共用
同一个 AgentPool + milu 全配默认 + 隔离设置。

```
平台（微信客服/飞书/Telegram…）
    │ 各自的回调/长轮询、加解密、消息格式
    ▼
Channel 适配器 ──规整成 InboundMessage──▶ dispatch（AgentRunner→milu Agent）
    ▲                                        │
    └──── 平台 API 回发 reply.text ◀── OutboundMessage ┘
```

- **Channel**（`milu.channels.*`）：一个平台一个适配器。webhook 型（微信客服/飞书）重写
  `register()` 挂回调路由；polling 型（Telegram）重写 `run()` 跑长轮询。
- **AgentRunner**（`milu.channels.runner`）：接 milu 的唯一处。持 `AgentPool`，把
  `InboundMessage` 跑过 per-用户 Agent，取 `AgentDone` 文本回出。
- **Gateway**（`milu.channels.gateway`）：装配——建 FastAPI、挂 webhook 路由、起 polling
  后台任务、统一 `/healthz` 与生命周期。
- **StateStore**（`milu.channels.state`）：游标 + 去重。`FileStateStore` 落
  `~/.milu/gateway/{channel}.json`，**重启不丢**（不重复回复、不漏拉/重放）。

## 一键启动：`milu gateway`

```bash
pip install "milu[gateway]"          # webhook 渠道需 cryptography；Telegram 无需
# 配好任一渠道的环境变量（见下），然后：
milu gateway                         # 按已配置的凭证自动启用对应渠道（默认 0.0.0.0:8800）
milu gateway --channel telegram      # 只启用指定渠道（逗号分隔）
milu gateway -p qwen -m qwen-plus --port 8800
milu gateway --no-persist            # 去重/游标用内存版（重启即丢）
```

- 默认操作模式 `auto`（无人值守自主决策）；`--mode` 可改。
- LLM 厂商/模型、运行限额、**隔离策略**全部读分层配置（`~/.milu/config.json` ←
  项目 `config/milu.json`）。多用户**工作区按 user_id 隔离**（`~/.milu/workspace/{user}`）。
- 启动横幅会打印启用的渠道、回调路径与隔离状态。

> 开发期二次开发/讲解接线，见 `examples/gateway_multi_channel.py`（手写 Runner + Gateway）。

## 各渠道配置

### 微信客服（经企业微信）
见 [`微信客服接入教程.md`](微信客服接入教程.md)。环境变量（`.env` 3.5 节）：
`WECHAT_KF_CORP_ID / WECHAT_KF_SECRET / WECHAT_KF_TOKEN / WECHAT_KF_AESKEY`，
回调路径默认 `/wechat/kf/callback`。**域名备案主体须 = 企业微信认证主体**，且自建应用
要配**企业可信IP** = 服务器出口 IP。

### 飞书（Lark）
1. 飞书开放平台「开发者后台 → 创建企业自建应用」，拿 **App ID / App Secret**。
2. 「权限管理」开通：接收消息 `im:message`、发送消息 `im:message:send_as_bot`（或
   `im:message`），并发布应用版本。
3. 「事件订阅」：
   - 请求地址填 `https://<你的域名>/feishu/event`；
   - 记下 **Verification Token**；如启用加密则记 **Encrypt Key**；
   - 订阅事件 **接收消息 `im.message.receive_v1`**。
4. 环境变量（`.env` 3.6 节）：
   ```
   FEISHU_APP_ID=cli_xxx
   FEISHU_APP_SECRET=xxx
   FEISHU_VERIFY_TOKEN=xxx
   FEISHU_ENCRYPT_KEY=xxx        # 启用事件加密时填，否则留空
   # FEISHU_API_BASE=https://open.feishu.cn   # 国际版 Lark 用 open.larksuite.com
   ```
5. 启动 `milu gateway` 后，回到「事件订阅」保存请求地址——milu 会自动回 `challenge`
   完成握手。之后在飞书里私聊机器人即可对话。

**两种接入模式**（`FEISHU_MODE`）：
- `webhook`（默认）：飞书 **push** 到你的回调 URL。需公网 HTTPS（生产用）。
- `ws`（**长连接**）：本进程**主动连飞书**收事件，**无需公网回调/HTTPS/隧道**，最适合
  **本地开发调试**（和 Telegram 一样省事）。设 `FEISHU_MODE=ws` 即可，发消息仍走
  tenant_access_token。需装 SDK：`pip install "milu[feishu-ws]"`（lark-oapi）。
  本地开发：在飞书后台「事件订阅 → 订阅方式」选「长连接」，然后：
  ```bash
  set FEISHU_MODE=ws            # PowerShell: $env:FEISHU_MODE="ws"
  milu gateway --channel feishu
  ```
  两种模式 `name` 同为 `feishu`，同一用户身份/会话连续一致；生产切回 webhook 不影响历史。

### Telegram
1. 找 **@BotFather** `/newbot` 创建机器人，拿 **token**。
2. 环境变量（`.env` 3.7 节）：
   ```
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   # TELEGRAM_API_BASE=https://api.telegram.org   # 国内服务器连不上官方域名时，
   #                                               # 指向代理或自建 Bot API Server
   ```
3. Telegram 走**长轮询**（无需公网回调/HTTPS），`milu gateway` 启动即开始收发。
   ⚠️ 国内云服务器通常连不上 `api.telegram.org`，需 `TELEGRAM_API_BASE` 指向可达地址。

## 多用户隔离与安全（公网部署务必看）

公网客服会把 `python_repl`/`shell_command`/文件工具暴露给陌生用户，必须隔离：

- **`milu config set multiuser strict`**：一键打开严格隔离档——代码进**断网 docker
  容器**（`sandbox.backend=docker`）+ 文件工具**关进工作区**（`agent.workspace_jail=true`）。
  `milu gateway` 会按此构造每个用户的 Agent（工作区还按 user_id 二次隔离）。需本机装 Docker。
- 无法用 docker 又想部分隔离：`multiuser=normal` + 手动
  `sandbox.backend=subprocess` + `agent.workspace_jail=true`。
- 密钥只放 `.env`（gitignore），不进 config.json；docker 容器不传宿主 env，密钥天然不可见。

## 从手写 run.py 迁移

早期微信客服用手写 `run.py`（`AgentPool.from_llm` + `on_text` 接线）。迁到网关：

```bash
# 旧：python run.py（手写 on_text 接 AgentPool）
# 新：
milu config set multiuser strict      # 若要 docker 强隔离（同旧 run.py 的 agent_kwargs）
milu gateway                          # 自动启用微信客服渠道，等价且更省心
```

systemd 单元把 `ExecStart` 从 `python run.py` 换成 `milu gateway` 即可（环境变量不变）。
回调路径、企业微信配置、nginx 反代都无需改动。

## 二次开发

```python
from milu.channels import AgentRunner, Gateway, FileStateStore
from milu.channels.feishu import FeishuChannel, FeishuConfig
from milu.llm.providers import ModelRegistry

llm = ModelRegistry.create("qwen", model="qwen-plus")
runner = AgentRunner.from_llm(llm, agent_kwargs={"sandbox": "docker", "workspace_jail": True})
store = FileStateStore()
channels = [FeishuChannel(FeishuConfig.from_env(), state=store)]
Gateway.from_runner(runner, channels).run(port=8800)
```

**自定义新平台**：继承 `milu.channels.base.Channel`，实现 `name` + `register`（webhook）
或 `run`（polling），在里面把平台消息组成 `InboundMessage`、`await dispatch(msg)`、再用
平台 API 发 `reply.text`。dispatch 一侧完全平台无关，无需改动 Runner/Gateway。
