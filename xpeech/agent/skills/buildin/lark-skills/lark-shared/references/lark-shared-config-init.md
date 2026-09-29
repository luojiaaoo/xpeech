# lark-cli 配置

应用凭据由 Xpeech 的 token-manager 服务配置并持有。不要运行 lark-cli 的配置初始化或登录命令，也不要创建 profile。

首次使用用户身份时执行 `lark-cli-auth login --scope "<本次任务所需 scope>"` 或 `lark-cli-auth login --domain <domain>` 获取设备授权链接；至少传入一个 `--scope` 或 `--domain`，token-manager 会在飞书返回的有效期内后台轮询。普通业务继续直接执行原生 `lark-cli` 命令。服务状态可用 `lark-cli-auth status` 查看。
