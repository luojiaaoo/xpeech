---
name: lark-skills
version: v1.0.90
description: "飞书/Lark 全能力聚合路由（基于 lark-cli）。涵盖审批、妙搭应用开发、考勤、多维表格 Base、日历、通讯录、云文档 Docx/Wiki、云盘 Drive、事件订阅、即时通讯 IM、邮箱、Markdown、视频会议与纪要/妙记、OKR、原生 OpenAPI、电子表格 Sheets、幻灯片 Slides、任务 Task、画板 Whiteboard、知识库 Wiki，以及会议纪要汇总、日程待办摘要两个工作流和 Skill 制作器。当用户有任一飞书/Lark 需求（发消息、查日程、读写文档/表格/幻灯片/Base、管理云盘、处理审批/任务/考勤、查会议记录、开发妙搭应用等）时使用本技能，按本文件目录读取对应子技能库的 SKILL.md 后执行。认证/登录/令牌/scope 等授权类需求走独立技能 lark-oauth，不在本技能内。"
---

# lark-skills — 飞书/Lark 能力聚合路由

本技能是 **`lark-*` 子技能库的聚合入口 / 路由**。它本身不包含具体操作指令，只负责把需求路由到正确的子技能库；每个能力的具体用法都在对应的子技能库 `SKILL.md` 里。

> 本技能刻意**不包含** `lark-oauth`（认证 / 登录 / 令牌刷新 / scope 申请等授权操作）。授权类需求走独立的顶级技能 `lark-oauth`。

## 使用方法（路由规则）

1. 收到飞书/Lark 相关需求后，在下方「目录」里按需求匹配最合适的子技能库。
2. 用 Read 工具读取该子技能库的 `SKILL.md`，按其指引执行。
3. 涉及认证、身份切换、权限不足（`missing_scopes`）、输出契约、高风险操作等通用规则时，先读取 `lark-shared/SKILL.md`。
4. 涉及登录 / 授权 / 刷新令牌 / 追加 scope 的需求，改走 `lark-oauth`（独立技能，不在本技能内）。

## 目录（子技能库一览）

```
lark-skills/
├── SKILL.md                        ← 本路由文件
├── lark-shared/                    共享底座：认证、身份、权限、输出契约、高风险操作（其余子技能库普遍依赖它）
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

## 路由速查（高频需求 → 子技能库）

| 用户想要… | 读取 |
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
| 认证 / 登录 / 令牌 / scope | `lark-oauth`（独立技能，不在此处） |
| 认证、身份切换、权限通用规则 | `lark-shared/SKILL.md` |

## 重要：相对路径说明

- 每个子技能库（如 `lark-approval/`）内部出现的**相对路径，都是相对该子技能库自己的目录**而言的——不是相对本路由 `lark-skills/SKILL.md`，也不是相对仓库根目录。
- 例如 `lark-approval/SKILL.md` 里写的 `../lark-shared/SKILL.md`，实际指向 `lark-skills/lark-shared/SKILL.md`（因为 `lark-approval` 与 `lark-shared` 都是 `lark-skills/` 下的同级目录）。
- 同理，子技能库 `references/`、`scripts/`、`scenes/` 等子目录里的 `../../xxx` 相对路径，也是从该文件自己所在的子目录逐级向上解析，最终仍落在 `lark-skills/` 内。
- 总之：**看到一个相对路径时，先确定它写在哪个子技能库的哪个文件里，再以那个文件所在目录为基准去解析。**
