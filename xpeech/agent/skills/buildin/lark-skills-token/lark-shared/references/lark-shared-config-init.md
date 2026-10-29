# lark-cli 配置

应用 ID 由 Xpeech 在构建时从 conf.toml 注入。用户访问令牌由调用方放入 /chat 的 Bearer JWT payload 的 `lark_user_access_token` 字段；后端验证后自动注入 shell 环境变量 `LARKSUITE_CLI_USER_ACCESS_TOKEN`。

不要运行 CLI 配置初始化或登录命令，不要创建 profile，也不要读取或打印令牌。普通业务直接执行 `lark-cli ... --as user`。令牌缺失、失效或权限不足时，报告问题，由调用方在新的 JWT 中提供有效令牌。
