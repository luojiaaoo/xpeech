---
name: lark-shared
version: 1.1.0
description: "Use for lark-cli identity, permissions, missing scopes, JWT user access tokens, and handling _notice JSON."
metadata:
  requires:
    bins: ["lark-cli"]
---

# lark-cli 共享规则

所有 `lark-*` skill 共享的底座：身份、认证、输出契约与高风险操作。

## 通用准则

1. **调用前先确认用法**：执行前读对应 reference 或跑 `--help`，别猜 flag 盲调。

2. **身份决定你代表谁操作**：`--as user` 代表用户本人（能看到、也能操作其日历、云空间/云盘/云存储等个人资源），`--as bot` 代表应用自己，应用级操作，只能访问bot自己的资源，bot 查用户资源会返回空成功而非报错。动手前先搞清楚身份`identity`。当前部署仅支持 `--as user`，令牌由请求 JWT 注入，不提供 bot token。身份模型和权限管理 → [`lark-shared-identity-and-permissions.md`](references/lark-shared-identity-and-permissions.md)。

3. **认证由调用方提供**：使用请求 JWT 注入的 `LARKSUITE_CLI_USER_ACCESS_TOKEN`，不要读取或打印令牌，不要主动登录、发起授权或获取新令牌。令牌缺失、失效或缺少 scope 时，报告缺失项，让调用方在新的 JWT 中提供具备所需权限的有效令牌。

4. **`--format json`（默认）下，判断成功用 `ok == true`（或进程退出码 0），不要用 `code == 0`**：成功信封没有顶层 `code` / `msg` 字段，`code` 只出现在错误信封的 `error` 内。按 OpenAPI 老格式 `{"code": 0, "msg": "ok"}`判断会把所有成功调用误判为失败——封装写入类命令时尤其危险。JSON 输出契约 → [`lark-shared-output-contract.md`](references/lark-shared-output-contract.md)。


## 安全规则

1. **禁止输出密钥**（appSecret、accessToken等）到终端明文。

2. **写入/删除操作前必须确认用户意图**。

3. 目标命令支持 `--dry-run` 时，用 `--dry-run` 预览危险请求。

4. **退出码 10 是高风险确认门禁（`risk: "high-risk-write"`），不是错误**：停下 → **向用户确认**（展示 `action`、`risk` 和关键参数）→ 取得**用户显式同意**后，将 `hint` 指出的确认 flag **追加到你原始 argv 的末尾**后重试；**绝不**静默加确认 flag 绕过 → [`lark-shared-high-risk-approval.md`](references/lark-shared-high-risk-approval.md)。

5. **文件路径只接受相对路径**：`--file`、`--output`、`--output-dir`、`@file` 等路径参数只接受 cwd 下的相对路径，传绝对路径会报 `unsafe file path`。数据输入（`@file`、大 JSON）优先用 stdin 传入，避免路径和转义问题。


## Reference 强触发索引

命中任一触发条件时，**MUST 在执行下一步前读取对应 reference**。命中多条时按表中顺序读取，同一reference只读取一次。

| 强触发条件（命中任一即必读） | Reference |
|---|---|
| 查看自己是谁(user/bot)、获取当前身份详细字段信息、身份诊断、`--as`选择逻辑、身份延续、登录态、认证、scope、授权和权限管理、`missing_scopes` 或 `console_url` | [`lark-shared-identity-and-permissions.md`](references/lark-shared-identity-and-permissions.md) |
| 需要依赖 JSON 输出契约判断成功 / 失败、读取 stdout / stderr，或为命令编写脚本与封装 | [`lark-shared-output-contract.md`](references/lark-shared-output-contract.md) |
| 准备执行high-risk-write(高风险操作)、判断命令风险等级、遇到退出码 exit 10、`confirmation_required`、确认后重试 | [`lark-shared-high-risk-approval.md`](references/lark-shared-high-risk-approval.md) |
| CLI 提示令牌缺失、配置缺失或 `config init --new` | [`lark-shared-config-init.md`](references/lark-shared-config-init.md) |
| 用户询问 notice、CLI版本更新、或输出含 `_notice`（升级 / skills 落后 / 废弃命令提示）| [`lark-shared-update-notice.md`](references/lark-shared-update-notice.md) |
