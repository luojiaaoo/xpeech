---
name: lark-cli-auth
description: 飞书用户授权管理。通过 token-manager 生成非阻塞设备授权链接、查看状态、注销授权和按 scope/domain 申请权限；不要使用 lark-cli 内置登录。
---

# 飞书授权管理

用户身份由独立 token-manager 管理。业务命令仍直接使用 `lark-cli`；授权命令只负责创建授权请求、查看状态和注销，不读取或展示 access token、refresh token 或应用密钥。

## 命令

```bash
lark-cli-auth login --scope "docs:doc:readonly drive:drive:readonly"
lark-cli-auth login --domain calendar,task
lark-cli-auth status
lark-cli-auth logout
```

`login` 必须明确传入本次任务需要的至少一个 `--scope` 或 `--domain`；两者可以组合。`--domain` 使用原生 lark-cli 的业务域目录，将该域对应的用户 scopes 展开后再提交授权；支持逗号、空格分隔和 `all`。缺少或传入空值会直接报错，不会发起授权。命令会立即输出一次性设备授权链接并退出。把完整链接原样交给用户；token-manager 会在飞书返回的有效期内后台轮询，用户完成授权后即可重跑原业务命令。客户端不要等待、轮询或要求用户回复“已完成”，也不要保存 device code。授权链接只显示地址和必要的 user code，不显示令牌。

新增权限时，执行 `login --scope` 或 `login --domain` 传入本次任务需要的权限；已授权范围由飞书授权服务维护，不由 manager 本地拼接。若业务命令返回 `authorization_required`，发送返回的 `authorization_url` 并结束当前任务；授权完成后重跑原命令。

## 状态与错误

`status` 只显示授权状态、绑定的飞书用户、scope 和过期时间。token-manager 不可用时报告服务不可用，等待服务恢复后重试。不要运行 `lark-cli auth login`、`lark-cli config`，也不要创建 profile 或手动修改 token 文件。

授权以当前 Xpeech session 为边界。不要把授权链接、scope 或状态跨 session 复用；不要并发为同一 session 创建多个授权请求。
