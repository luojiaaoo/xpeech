# 🚀 Xpeech

**下一代智能 AI Agent 交互平台**

> 让每一次对话都充满智慧，让每一个智能体都能理解你的真实意图。

---

## ✨ 项目愿景

Xpeech 致力于打造一个**开箱即用、灵活扩展、生产级**的 AI Agent 服务框架。我们相信：

- 🎯 **真正的智能不是单轮问答，而是持续的理解与行动**
- 🔌 **优秀的架构应该让开发者专注于业务，而非基础设施**
- 🌐 **多模态交互是未来的标配，而非奢侈品**

我们的目标是为开发者提供：
- ⚡ **极简接入**：5 分钟启动一个生产级 AI Agent API
- 🧩 **无限扩展**：插件化的 Provider 架构，无缝对接各类大模型
- 🛠️ **工具调用原生支持**：让 Agent 真正具备操作世界的能力
- 📊 **企业级可观测**：完善的会话追踪、元数据管理

---

## 📖 技术架构

### 核心特性

- **🏗️ 现代化技术栈**：Python 3.12+ | FastAPI | Pydantic | Uvicorn | LiteLLM
- **🤖 多模态支持**：文本、图片混合输入输出
- **🔀 Agent Loop**：内置智能体循环，自动处理工具调用链（最多30次迭代）
- **📡 标准 API 设计**：RESTful 接口，完整 OpenAPI 文档，支持 SSE 流式响应
- **🔐 会话管理**：基于工作区的会话隔离，支持多渠道接入
- **⚙️ 配置驱动**：TOML + 环境变量双层配置，灵活部署
- **🧠 记忆系统**：长期记忆（MEMORY.md）+ 历史摘要（HISTORY.md）+ 会话压缩
- **🛡️ 安全沙箱**：工作区路径限制 + 危险命令过滤 + 内网访问拦截

### 架构概览

```
┌─────────────────────────────────────────────────┐
│                  API Gateway                     │
│              (FastAPI + Uvicorn)                 │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
   │ Session │ │ Content │ │  File   │
   │ Manager │ │ Parser  │ │ Handler │
   └────┬────┘ └────┬────┘ └────┬────┘
        │            │            │
        └────────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │     Agent Loop Engine    │
        │  (工具调用 / 多轮推理)   │
        │  - 会话压缩策略          │
        │  - 记忆巩固机制          │
        │  - 最大迭代控制          │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    Provider Interface    │
        │  (LiteLLM 封装层)        │
        │  - 工具注册与管理        │
        │  - 多模型适配            │
        └────────────┬────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
   │  File   │ │  Shell  │ │ Memory  │
   │  Tools  │ │  Tools  │ │  Store  │
   └─────────┘ └─────────┘ └─────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.12
- uv (推荐包管理器)

### 安装依赖

```bash
uv sync
```

### 配置说明

项目使用双层配置系统：

1. **TOML 配置文件** (`conf.toml`)：定义路径等基础配置
2. **环境变量** (`.env`)：定义 API Key、模型地址等敏感信息

示例 `.env`：
```env
BASE_URL = https://ark.cn-beijing.volces.com/api/coding/v3
API_KEY = your_api_key_here
MODEL_NAME = glm-5.1
```

### 启动服务

```bash
python -m xpeech
```

服务默认运行在 `http://localhost:7878`

### API 文档

启动服务后访问：
- Swagger UI: `http://localhost:7878/docs`
- ReDoc: `http://localhost:7878/redoc`

---

## 💡 核心 API 示例

### 发送消息（支持多模态）

```bash
curl -X POST "http://localhost:7878/chat" \
  -F "session_id=user_123" \
  -F 'session_metadata={"sender_id": "user123", "channel": "web"}' \
  -F 'content=[{"text": "你好，请介绍一下自己"}]' \
  -F "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")"
```

### 上传文件并发送

```bash
curl -X POST "http://localhost:7878/chat" \
  -F "session_id=session_001" \
  -F 'session_metadata={"channel": "api"}' \
  -F 'content=[{"text": "帮我分析这张图片"}]' \
  -F "files=@image.png"
```

### Python 调用

```python
import requests

response = requests.post(
    "http://localhost:7878/chat",
    data={
        "session_id": "session_001",
        "session_metadata": '{"channel": "api"}',
        "content": '[{"text": "帮我分析这张图片"}]',
    },
    files={"files": open("image.png", "rb")}
)

# SSE 流式响应处理
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### SSE 事件类型

服务端推送的事件包括：
- `thinking`: AI 思考过程
- `assistant`: 助手回复内容
- `tool_call`: 工具调用信息
- `tool_call_result`: 工具执行结果
- `command`: 系统命令响应

---

## 🏗️ 项目结构

```
xpeech/
├── xpeech/
│   ├── agent/
│   │   ├── loop.py              # Agent 循环引擎（工具调用链处理、会话压缩、记忆巩固）
│   │   ├── memory.py            # 长期记忆存储管理
│   │   ├── prompt/
│   │   │   ├── system.py        # 系统提示词构建
│   │   │   └── helper.py        # 用户提示词构建
│   │   ├── server/
│   │   │   ├── api.py           # API 路由定义（/chat 端点）
│   │   │   ├── app.py           # FastAPI 应用入口
│   │   │   ├── schema.py        # 请求/响应数据模型
│   │   │   └── server.py        # 服务器配置
│   │   ├── skills/
│   │   │   └── buildin/memory/  # 内置技能（记忆管理）
│   │   └── tools/
│   │       ├── filesystem.py    # 文件系统工具（读/写/编辑/列表）
│   │       ├── shell.py         # Shell 命令执行工具（带安全防护）
│   │       └── helper.py        # 工具辅助函数
│   ├── config/
│   │   ├── settings.py          # 配置管理（Pydantic Settings）
│   │   └── helper.py            # TOML 配置加载器
│   ├── provider/
│   │   ├── litellm_provider.py  # LiteLLM Provider 实现
│   │   └── schema.py            # Provider 数据模型
│   ├── utils/
│   │   ├── session.py           # 会话工作区模板创建
│   │   ├── helper.py            # 通用辅助函数
│   │   └── security/
│   │       └── network.py       # 网络安全检查
│   └── __main__.py              # 入口文件
├── workspace_base/              # 工作区基目录（按 session_id 隔离）
│   └── {session_id}/
│       ├── AGENTS.md            # Agent 指令
│       ├── SOUL.md              # Agent 人格设定
│       ├── USER.md              # 用户信息
│       ├── memory/
│       │   ├── MEMORY.md        # 长期记忆
│       │   └── HISTORY.md       # 历史摘要
│       └── skills/              # 自定义技能
├── session/
│   └── history/                 # 会话历史记录（YAML 格式）
├── conf.toml                    # 配置文件
├── .env                         # 环境变量
├── pyproject.toml               # 项目配置
└── README.md
```

---

## 🧠 核心功能详解

### 1. Agent Loop 引擎

Agent Loop 是 Xpeech 的核心，负责处理多轮对话和工具调用：

- **自动工具调用链**：AI 可以连续调用多个工具完成任务
- **最大迭代控制**：默认最多 30 次迭代，防止无限循环
- **智能中断机制**：接近上限时自动添加停止提示

### 2. 会话压缩策略

当上下文超过阈值时，自动触发四级压缩：

1. **一级压缩**：截断超长工具执行结果（保留最近 4 次对话的完整结果）
2. **二级压缩**：按时间窗口保留消息（7/6/5/4/3/2 天递减）
3. **三级压缩**：AI 总结历史消息 + 保留最近 4 次对话
4. **四级压缩**：完全总结所有历史

### 3. 记忆系统

- **长期记忆**（`memory/MEMORY.md`）：持久化存储用户偏好、重要事实
- **历史摘要**（`memory/HISTORY.md`）：记录关键事件和决策
- **记忆巩固**：在 `/new` 命令或压缩时触发，AI 自动提取关键信息

### 4. 工具系统

#### 文件系统工具
- `read_file`: 读取文件内容
- `write_file`: 写入文件（自动创建父目录）
- `edit_file`: 精确替换文件内容
- `list_dir`: 列出目录内容

#### Shell 工具
- 支持 Bash 命令执行
- **安全防护**：
  - 危险命令过滤（rm -rf、format、shutdown 等）
  - 工作区路径限制（防止逃逸）
  - 内网访问拦截
  - 超时控制（60 秒）
  - 输出长度限制（10000 字符）

### 5. 工作区管理

每个会话拥有独立的工作区：
- 自动创建模板文件（AGENTS.md、SOUL.md、USER.md）
- 隔离的文件系统和内存空间
- 支持自定义技能和配置

### 6. 命令系统

支持的斜杠命令：
- `/help`: 显示帮助信息
- `/new`: 新建会话（触发记忆巩固并清空历史）

---

## 🔮 路线图

- [x] ✅ 基础 FastAPI 服务框架
- [x] ✅ 多模态输入支持（文本 + 图片）
- [x] ✅ 会话元数据管理
- [x] ✅ Agent Loop 实现（工具调用自动循环）
- [x] ✅ LiteLLM Provider 集成
- [x] ✅ 流式响应支持 (SSE)
- [x] ✅ 上下文窗口管理与压缩
- [x] ✅ 插件化工具注册系统
- [x] ✅ 会话持久化存储（YAML）
- [x] ✅ 长期记忆系统
- [ ] 🔄 Prometheus 指标导出
- [ ] 🔄 更多内置工具（搜索、代码执行等）
- [ ] 🔄 多渠道适配器（飞书、微信、Telegram）
- [ ] 🔄 Web UI 界面
- [ ] 🔄 技能市场与插件系统

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是：

- 🐛 Bug 报告
- 💡 功能建议
- 📖 文档改进
- 🔧 代码贡献

请提交 Issue 或 Pull Request。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-org/xpeech.git
cd xpeech

# 安装依赖
uv sync

# 运行测试
pytest test/

# 代码格式化
ruff format .
ruff check .
```

---

## 📄 License

MIT License

---

<div align="center">

**用代码构建智能，用对话连接未来** 🌟

Made with ❤️ by Xpeech Team

</div>
