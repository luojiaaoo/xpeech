# 身份与权限

## 认证任务速查

| 用户意图 | 命令 |
|---|---|
| 发起用户授权 | `lark-cli-auth login --scope "<本次任务所需 scope>"` |
| 按业务域发起用户授权 | `lark-cli-auth login --domain calendar,task` |
| 按 scope 增量授权 | `lark-cli-auth login --scope "calendar:calendar:readonly docs:doc:readonly"` |
| 查看当前授权用户与有效期 | `lark-cli-auth status` |
| 注销当前 session 授权 | `lark-cli-auth logout` |
| bot 缺少权限 | 将错误中的 `console_url` 原样提供给用户，请管理员在飞书后台开通 scope |

`lark-cli-auth login --scope "<本次任务所需 scope>"` 或 `lark-cli-auth login --domain <domain>` 立即返回设备授权 URL。至少需要一个 `--scope` 或 `--domain`；两者可组合。`--domain` 按原生 lark-cli 的业务域目录展开为用户 scopes，支持逗号/空格分隔和 `all`。把 URL 原样发送给用户并结束当前授权步骤；token-manager 在飞书返回的有效期内后台轮询，授权完成后重跑原业务命令。客户端不轮询、不要求用户回复确认，也不展示 token。

## 身份类型

| 身份 | 标识 | 获取方式 | 适用场景 |
|---|---|---|---|
| user 用户身份 | `--as user` | token-manager 中的当前 session 授权 | 访问用户自己的日历、云盘、文档和邮箱 |
| bot 应用身份 | `--as bot` | token-manager 的应用 token | 应用自身资源和机器人操作 |

业务命令仍使用原生 `lark-cli` 参数。当前 session 是授权隔离边界；不要把其他 session 的授权 URL 或状态复用过来。

## 权限不足处理

错误中的 `missing_scopes` 是用户身份需要补充的范围。按错误中的 scope 执行一次授权：

```bash
lark-cli-auth login --scope "scope.one scope.two"
```

bot 身份缺少权限时，不执行用户授权；使用错误中的 `console_url` 引导管理员在飞书开发者后台开通应用权限。

## 输出安全

不要读取、打印或手动修改 token-manager 数据库、access token、refresh token 或 app secret。`status` 只返回用户、scope、状态和过期时间。
