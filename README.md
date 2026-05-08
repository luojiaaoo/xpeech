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

- **🏗️ 现代化技术栈**：Python 3.12+ | FastAPI | Pydantic | Uvicorn
- **🤖 多模态支持**：文本、图片、文件混合输入输出
- **🔀 Agent Loop**：内置智能体循环，自动处理工具调用链
- **📡 标准 API 设计**：RESTful 接口，完整 OpenAPI 文档
- **🔐 会话管理**：支持多渠道（飞书、微信、Web）接入
- **⚙️ 配置驱动**：环境变量优先，灵活部署

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
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    Provider Interface    │
        │  (OpenAI / 自定义模型)   │
        └─────────────────────────┘
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

### 启动服务

```bash
python -m xpeech
```

服务默认运行在 `http://localhost:8000`

### API 文档

启动服务后访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 💡 核心 API 示例

### 发送消息

```bash
curl -X POST "http://localhost:8000/chat" \
  -F "session_id=user_123" \
  -F 'session_metadata={"sender_id": "user123", "channel": "web"}' \
  -F 'content={"content": [{"text": "你好，请介绍一下自己"}]}' \
  -F "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")"
```

### Python 调用

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    data={
        "session_id": "session_001",
        "session_metadata": '{"channel": "api"}',
        "content": '{"content": [{"text": "帮我分析这张图片"}]}',
    },
    files={"files": open("image.png", "rb")}
)

print(response.json())
```

---

## 🏗️ 项目结构

```
xpeech/
├── xpeech/
│   ├── agent/
│   │   ├── loop.py          # Agent 循环引擎（工具调用链处理）
│   │   └── server/
│   │       ├── api.py       # API 路由定义
│   │       ├── app.py       # FastAPI 应用入口
│   │       ├── schema.py    # 请求/响应数据模型
│   │       └── server.py    # 服务器配置
│   ├── provider/
│   │   ├── chat.py          # LLM Provider 实现
│   │   └── schema.py        # Provider 数据模型
│   └── __main__.py          # 入口文件
├── pyproject.toml           # 项目配置
└── README.md
```

---

## 🔮 路线图

- [x] ✅ 基础 FastAPI 服务框架
- [x] ✅ 多模态输入支持（文本 + 图片）
- [x] ✅ 会话元数据管理
- [ ] 🔄 Agent Loop 实现（工具调用自动循环）
- [ ] 🔄 OpenAI Compatible Provider
- [ ] 📝 流式响应支持 (SSE)
- [ ] 🧠 上下文窗口管理
- [ ] 🔧 插件化工具注册系统
- [ ] 💾 会话持久化存储
- [ ] 📊 Prometheus 指标导出
- [ ] 🌐 多渠道适配器（飞书、微信、Telegram）

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是：

- 🐛 Bug 报告
- 💡 功能建议
- 📖 文档改进
- 🔧 代码贡献

请提交 Issue 或 Pull Request。

---

## 📄 License

MIT License

---

<div align="center">

**用代码构建智能，用对话连接未来** 🌟

Made with ❤️ by Xpeech Team

</div>