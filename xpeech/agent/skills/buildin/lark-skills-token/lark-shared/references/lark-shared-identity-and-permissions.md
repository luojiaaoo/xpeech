# 身份与权限

## 认证方式

当前部署仅支持 `--as user`。调用方在 /chat 的 Bearer JWT payload 中携带 `lark_user_access_token`，后端校验 JWT 后按请求注入 `LARKSUITE_CLI_USER_ACCESS_TOKEN`。CLI 直接使用此令牌，不主动授权、获取或刷新令牌。

不同请求的令牌互相隔离，同一 session 的下一次请求也需重新携带。不要把令牌写入会话记录、文件或模型消息，不要读取、打印或手动修改令牌和应用密钥。

## 缺失、失效与权限不足

- 缺少或过期的令牌：报告缺失或失效，由调用方在新的 JWT 中提供有效令牌。
- 缺少权限：保留错误中的 `missing_scopes` 或所需 scope，交由调用方处理后提供具备相应权限的令牌。
- API 仅支持 bot 身份：当前部署没有 tenant access token，说明该操作无法使用当前用户令牌完成。
- 不执行任何 CLI 登录、配置初始化或 OAuth 授权命令，也不生成授权链接。
