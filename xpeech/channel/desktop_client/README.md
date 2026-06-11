# Xpeech 桌面客户端

这是 Xpeech 的独立桌面客户端。

- 前端：Vite + React + antd + Ant Design X
- 桌面壳：pywebview
- 前后端通信：pywebview JS-Python bridge
- 依赖服务：已经启动的 Xpeech API

桌面客户端是独立客户端，不挂在 `python -m xpeech` 主入口下面，后续也按独立应用打包。

## 安装依赖

在仓库根目录安装 Python 依赖：

```bash
uv sync
```

安装前端依赖：

```bash
cd xpeech/channel/desktop_client/frontend
npm install
```

## 启动 Xpeech API

桌面客户端只连接已有的 Xpeech API，不会自己启动 API 服务。

在仓库根目录执行：

```bash
uv run -m xpeech api
```

默认 API 地址：

```text
http://127.0.0.1:7878
```

## 开发模式启动

先启动 Vite 开发服务器：

```bash
cd xpeech/channel/desktop_client/frontend
npm run dev
```

再打开另一个终端，在仓库根目录启动桌面壳：

```bash
uv run python -m xpeech.channel.desktop_client --dev
```

如果要指定 API 地址：

```bash
uv run python -m xpeech.channel.desktop_client --dev --api-base-url http://127.0.0.1:7878
```

## 生产模式启动

先构建前端：

```bash
cd xpeech/channel/desktop_client/frontend
npm run build
```

再回到仓库根目录启动桌面客户端：

```bash
uv run python -m xpeech.channel.desktop_client
```

如果要指定 API 地址：

```bash
uv run python -m xpeech.channel.desktop_client --api-base-url http://127.0.0.1:7878
```

## 配置文件

桌面客户端配置文件：

```text
xpeech/channel/desktop_client/config.toml
```

默认内容：

```toml
api_base_url = "http://127.0.0.1:7878"
```

桌面 UI 里也可以修改这个 API 地址。

## 会话身份

桌面客户端的会话 ID 格式：

```text
desktop_{machine_code}
```

其中 `machine_code` 来自 `uuid.getnode()`，格式化为 12 位小写十六进制字符串。

桌面客户端会把系统登录用户名作为 `sender_name` 发送给 Xpeech API。

## 打包说明

桌面客户端独立打包时使用这个入口：

```text
xpeech.channel.desktop_client.__main__:main
```

打包前先构建前端：

```bash
cd xpeech/channel/desktop_client/frontend
npm run build
```

打包产物需要包含：

```text
xpeech/channel/desktop_client/frontend/dist
xpeech/channel/desktop_client/config.toml
```

不要把桌面客户端挂到仓库级的 `xpeech.__main__` 主 CLI 上，也不要依赖 `uv run -m xpeech desktop` 这种启动方式。
