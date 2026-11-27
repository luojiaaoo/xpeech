---
name: lark-skills
description: "飞书/Lark 全能力聚合路由（基于 lark-cli）。涵盖审批、妙搭应用开发、考勤、多维表格 Base、日历、通讯录、云文档 Docx/Wiki、云盘 Drive、事件订阅、即时通讯 IM、邮箱、Markdown、视频会议与纪要/妙记、OKR、原生 OpenAPI、电子表格 Sheets、幻灯片 Slides、任务 Task、画板 Whiteboard、知识库 Wiki，以及会议纪要汇总、日程待办摘要两个工作流和 Skill 制作器。当用户有任一飞书/Lark 需求时使用本技能，按任务通过 read_file 读取对应参考文档后执行。"
---

# lark-skills — 飞书/Lark 能力聚合路由

本技能将飞书/Lark 的业务能力整合在一个入口中。各业务目录下的 `SKILL.md` 是本技能的参考文档，保留原始文件名便于维护，不作为独立技能注册，也不通过 `read_skill` 加载。

> 本技能负责业务能力路由；授权、状态和 scope 申请由独立的 `lark-cli-auth` 技能处理。

## 使用方法（路由规则）

1. 首先通过 `read_file` 读取 `lark-shared/SKILL.md`，遵守其中的身份、认证、权限、输出契约和高风险操作规则。
2. 按下方速查表选择业务参考文档，通过 `read_file` 按路径读取，例如 `<本技能目录>/lark-im/SKILL.md`；再按任务读取文档中引用的详细资料。
3. 参考文档之间的 `SKILL.md` 链接同样通过 `read_file` 读取，不按 YAML 中的 `name` 调用 `read_skill`。文档较长时分段读取所需内容，标明必读的内容必须读完。
4. 普通业务直接执行 `lark-cli`。遇到 `authorization_required` 时读取并原样发送其中的授权链接；授权、注销和追加 scope 通过 `read_skill(name="lark-cli-auth")` 加载独立的认证技能。

## 目录（业务参考文档）

```
lark-skills/
├── SKILL.md                        ← 本路由文件
├── lark-shared/                    共享底座：认证、身份、权限、输出契约、高风险操作（各业务文档共用）
├── lark-approval/                  审批：待办/已办/实例查询、搜索并发起审批
├── lark-apps/                      妙搭（Spark/Miaoda）应用：创建/开发/部署/发布、创意设计、环境变量、自动化触发器
├── lark-attendance/                考勤：打卡记录查询
├── lark-base/                      多维表格 Base：表/字段/记录/视图/统计/公式/表单/仪表盘/AppMode/模板中心
├── lark-calendar/                  日历：日程管理、会议室预订、忙闲/推荐时段
├── lark-contact/                   通讯录：名字/邮箱 ↔ open_id 解析，机器人/智能体搜索
├── lark-doc/                       云文档：读取/创建/编辑 Docx、Wiki、思维笔记，图片附件
├── lark-drive/                     云盘：文件/文件夹管理、上传下载、评论/权限、导入转换、密级标签
├── lark-event/                     事件订阅：实时事件流消费（IM/审批/任务/会议/纪要/画板等）
├── lark-im/                        即时通讯：收发消息、群聊、图片文件、表情、加急、交互卡片、卡片回调
├── lark-mail/                      邮箱：草稿/发送/回复/转发/搜索/文件夹/标签/联系人/收信规则
├── lark-markdown/                  Markdown：查看/创建/上传/编辑/比较 Markdown 文件
├── lark-meeting/                   视频会议：会议记录、纪要/逐字稿/妙记、实时会议问答、会中聊天
├── lark-minutes/                   兼容入口：统一交由 lark-meeting 处理
├── lark-note/                      兼容入口：统一交由 lark-meeting 处理
├── lark-okr/                       OKR：周期/目标/关键结果/对齐/进展管理
├── lark-openapi-explorer/          原生 OpenAPI：挖掘未经 CLI 封装的原生接口
├── lark-sheets/                    电子表格：创建/编辑表格、单元格、公式、图表、透视表、条件格式等
├── lark-skill-maker/               Skill 制作器：把飞书 API 操作封装为可复用 Skill
├── lark-slides/                    幻灯片：创建/编辑演示文稿、页面管理
├── lark-task/                      任务：待办、清单、子任务、协作、附件、任务智能体
├── lark-vc/                        兼容入口：统一交由 lark-meeting 处理
├── lark-vc-agent/                  兼容入口：统一交由 lark-meeting 处理
├── lark-whiteboard/                画板：查询/导出/更新文档中的画板
├── lark-wiki/                      知识库：空间/成员/节点管理
├── lark-workflow-meeting-summary/  工作流：汇总指定时间范围的会议纪要并生成报告
└── lark-workflow-standup-report/   工作流：日程 + 任务待办摘要
```

## 路由速查（高频需求 → 参考文档）

| 用户想要… | 参考文档（相对本技能目录） |
|---|---|
| 发消息、搜聊天记录、管群、发卡片 | `lark-im/SKILL.md` |
| 查/建日程、订会议室 | `lark-calendar/SKILL.md` |
| 读写云文档 / 思维笔记 | `lark-doc/SKILL.md` |
| 表格数据处理 / 建模 | `lark-sheets/SKILL.md` |
| 多维表格 / BaseApp | `lark-base/SKILL.md` |
| 云盘文件整理 / 上传下载 / 导入转换 | `lark-drive/SKILL.md` |
| 会议记录 / 妙记 / 逐字稿 / 会中问答 | `lark-meeting/SKILL.md` |
| 审批待办 / 发起审批 | `lark-approval/SKILL.md` |
| 待办任务 / 清单 | `lark-task/SKILL.md` |
| 演示文稿 | `lark-slides/SKILL.md` |
| 知识库结构 / 成员 | `lark-wiki/SKILL.md` |
| 考勤打卡记录 | `lark-attendance/SKILL.md` |
| OKR 目标与进展 | `lark-okr/SKILL.md` |
| 通讯录找人 / open_id 反查 | `lark-contact/SKILL.md` |
| 邮箱 | `lark-mail/SKILL.md` |
| Markdown 文件 | `lark-markdown/SKILL.md` |
| 开发 / 部署妙搭应用、创意设计 | `lark-apps/SKILL.md` |
| 实时事件流 / 机器人回调 | `lark-event/SKILL.md` |
| 现有 CLI 覆盖不了的接口 | `lark-openapi-explorer/SKILL.md` |
| 封装新的飞书 Skill | `lark-skill-maker/SKILL.md` |
| 会议纪要周报 | `lark-workflow-meeting-summary/SKILL.md` |
| 今日/本周日程与待办摘要 | `lark-workflow-standup-report/SKILL.md` |
| 认证 / 登录 / 令牌 / scope | `lark-cli-auth`（独立技能，不在此处） |
| 认证、身份切换、权限通用规则 | `lark-shared/SKILL.md` |

## 重要：相对路径说明

- 先以 `read_skill` 返回的本技能目录为基准，解析速查表中的参考文档路径。
- 业务参考文档（如 `lark-approval/SKILL.md`）正文中的相对链接，以该文档所在目录为基准解析；例如其 `references/xxx.md` 对应 `<本技能目录>/lark-approval/references/xxx.md`。
- 例如 `lark-approval/SKILL.md` 里写的 `../lark-shared/SKILL.md`，实际指向 `lark-skills/lark-shared/SKILL.md`（因为 `lark-approval` 与 `lark-shared` 都是 `lark-skills/` 下的同级目录）。
- 同理，业务目录下 `references/`、`scripts/`、`scenes/` 等子目录里的 `../../xxx` 相对路径，也是从该文件自己所在的子目录逐级向上解析，最终仍落在 `lark-skills/` 内。
- 总之：**看到一个相对路径时，先确定它写在哪个业务目录的哪个文件里，再以那个文件所在目录为基准去解析。**
