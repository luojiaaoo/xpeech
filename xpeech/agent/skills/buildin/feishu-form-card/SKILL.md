---
name: feishu-form-question-card
description: 当用户需求信息不足、需要通过表单向用户提问并收集结构化信息时，生成飞书卡片 2.0 表单 JSON。Skill 本体负责整体卡片框架，具体字段组件写法参考 references。
---

# 飞书表单追问卡片

- Skill 路径引用均采用**相对本 skill 根目录**的形式（`references/xxx.md`）——agent 或用户按自身安装位置解析，不依赖任何绝对路径。

## 作用

当用户提出一个需求，但当前信息不足以继续完成任务时，使用本 Skill 生成一个飞书表单卡片，通过 joyride_request_human_input 向用户追问关键信息。

这个 Skill 不是让用户手动描述字段配置。

大模型需要根据用户的实际需求，自行判断：

1. 当前任务缺少哪些关键信息。
2. 哪些信息适合通过表单收集。
3. 每个问题应该使用哪种表单组件。
4. 选项类问题应提供哪些合理选项。
5. 生成完整可用的飞书卡片 JSON。

## 适用场景

当用户需求存在以下情况时，可使用本 Skill：

- 缺少执行任务所需的关键参数。
- 需要用户从多个选项中选择。
- 需要用户补充文本信息。
- 需要用户选择日期。
- 需要用户选择日期和时间。
- 一次性需要向用户追问多个结构化问题。

## 不适用场景

以下情况不要使用表单追问：

- 只缺少一个很简单的信息，直接自然语言提问更合适。
- 可以根据上下文合理推断，不需要继续询问。
- 用户已经给出完整信息。
- 问题不适合结构化收集，例如开放式讨论、观点交流、头脑风暴。

## 支持的提问组件

根据要收集的信息类型选择组件：

| 需要收集的信息 | 使用组件 |
|---|---|
| 用户只能选择一个答案 | `select_static` |
| 用户可以选择多个答案 | `multi_select_static` |
| 用户需要输入文本 | `plain_text` |
| 用户需要选择日期 | `date_picker` |
| 用户需要选择日期和时间 | `picker_datetime` |

组件详细写法参考：

- `references/select_static.md`
- `references/multi_select_static.md`
- `references/plain_text.md`
- `references/date_picker.md`
- `references/picker_datetime.md`

## 组件选择原则

### 单选问题

当问题只有一个答案时，使用 `select_static`。

例如：

- 请选择输出格式
- 请选择目标平台
- 请选择优先级
- 请选择语言

### 多选问题

当问题允许多个答案同时成立时，使用 `multi_select_static`。

例如：

- 请选择需要包含的模块
- 请选择目标渠道
- 请选择需要支持的功能
- 请选择通知对象

### 文本输入问题

当需要用户自由输入内容时，使用 `plain_text` 对应的输入框组件。

例如：

- 请输入项目名称
- 请输入补充说明
- 请输入目标描述
- 请输入联系人

注意：

- 用户侧类型称为 `plain_text`
- 最终飞书卡片组件使用 `"tag": "input"`

### 日期问题

当只需要日期时，使用 `date_picker`。

例如：

- 请选择开始日期
- 请选择截止日期
- 请选择入职日期

### 日期时间问题

当需要精确到时间时，使用 `picker_datetime`。

例如：

- 请选择会议时间
- 请选择提醒时间
- 请选择任务截止时间

## 追问设计原则

生成表单问题时遵循以下原则：

1. 只问完成任务所必需的信息。
2. 不要把已知信息重复问给用户。
3. 问题标题要清晰、简短、可直接作答。
4. 优先使用选择组件降低用户输入成本。
5. 如果可选项无法合理枚举，则使用文本输入框。
6. 单选与多选要严格区分。
7. 日期和日期时间要严格区分。
8. 表单问题数量保持克制，避免一次追问过多。
9. 如果需要多个字段，按用户完成任务的思考顺序排列。
10. 如果选项值未特别要求，使用简洁稳定的字符串值。

## 字段标题格式

每一个问题组件前，都必须先生成一个问题标题 `div`。

标题格式固定如下：

```json
{
  "tag": "div",
  "text": {
    "tag": "plain_text",
    "content": "{{问题标题}}",
    "text_size": "normal_v2",
    "text_align": "left",
    "text_color": "default"
  },
  "margin": "0px 0px 0px 0px"
}
```

其中：

- `{{问题标题}}` 是大模型根据当前需求生成的提问标题。
- 标题应是用户可以直接理解的问题或字段名。
- 示例：
  - `请选择输出格式`
  - `请选择需要包含的模块`
  - `请输入项目名称`
  - `请选择开始日期`
  - `请选择会议时间`

## 字段名称规则

每个组件都需要设置 `name`。

`name` 用于表单提交后识别用户回答。

生成规则：

1. 优先使用清晰、稳定、可读的字段名。
2. 不要使用含义不明的随机名称。
3. 若无额外约束，可根据问题含义生成英文蛇形命名。

示例：

| 问题标题 | name |
|---|---|
| 请选择输出格式 | `output_format` |
| 请选择目标平台 | `target_platform` |
| 请输入项目名称 | `project_name` |
| 请选择开始日期 | `start_date` |
| 请选择会议时间 | `meeting_time` |

## 卡片标题规则

卡片标题应表达“需要用户补充信息”。

推荐写法：

- `请补充需求信息`
- `请确认以下信息`
- `请完善任务参数`
- `请填写以下内容`

卡片副标题应简要说明补充信息的目的。

推荐写法：

- `补充以下信息后，我将继续处理`
- `请确认关键参数，以便继续生成结果`
- `请填写任务所需信息`

## 整体卡片外框架

生成飞书表单卡片时，使用以下整体框架，副标题中告知用户超时时间：

```json
{
  "schema": "2.0",
  "config": {
    "update_multi": true,
    "style": {
      "text_size": {
        "normal_v2": {
          "default": "normal",
          "pc": "normal",
          "mobile": "heading"
        }
      }
    }
  },
  "body": {
    "direction": "vertical",
    "padding": "12px 12px 12px 12px",
    "elements": [
      {
        "tag": "form",
        "elements": [
          "{{问题标题 div}}",
          "{{对应问题组件}}",
          "{{更多问题标题 div 与问题组件}}",
          "{{提交取消按钮}}"
        ],
        "padding": "4px 0px 4px 0px",
        "margin": "0px 0px 0px 0px",
        "name": "question_form"
      }
    ]
  },
  "header": {
    "title": {
      "tag": "plain_text",
      "content": "{{主标题}}"
    },
    "subtitle": {
      "tag": "plain_text",
      "content": "{{副标题}}"
    },
    "template": "wathet",
    "icon": {
      "tag": "standard_icon",
      "token": "thumbsup_filled"
    },
    "padding": "12px 12px 12px 12px"
  }
}
```

## 表单按钮

所有追问字段生成完成后，在 `form.elements` 最后追加以下按钮：

```json
{
  "tag": "column_set",
  "columns": [
    {
      "tag": "column",
      "width": "auto",
      "elements": [
        {
          "tag": "button",
          "text": {
            "tag": "plain_text",
            "content": "提交"
          },
          "type": "primary",
          "width": "default",
          "form_action_type": "submit",
          "name": "submit_button"
        }
      ],
      "vertical_align": "top"
    },
    {
      "tag": "column",
      "width": "auto",
      "elements": [
        {
          "tag": "button",
          "text": {
            "tag": "plain_text",
            "content": "取消"
          },
          "type": "default",
          "width": "default",
          "form_action_type": "reset",
          "name": "reset_button"
        }
      ],
      "vertical_align": "top"
    }
  ]
}
```

## 生成步骤

当决定使用表单向用户追问时：

1. 分析用户当前需求。
2. 找出继续执行前必须确认的信息。
3. 删除已经从上下文中明确获得的信息。
4. 为每个缺失信息选择最合适的组件。
5. 为选择组件补充合理选项。
6. 为每个问题生成标题 `div`。
7. 根据 `references` 中的对应组件模板生成字段组件。
8. 将所有问题加入同一个 `form.elements`。
9. 在末尾追加提交和取消按钮。
10. 输出完整飞书卡片 JSON。

## 输出要求

默认只输出完整 JSON。

不要输出：

- 字段设计过程
- 组件选择解释
- Markdown 代码块外的额外说明
- 给用户看的字段配置说明

除非调用方明确要求解释，否则最终结果应直接是飞书卡片 JSON。

## 示例理解方式

### 用户需求

用户说：

> 帮我生成一个活动报名页面。

### 大模型判断

当前可能缺少：

- 活动名称
- 报名截止日期
- 报名页面风格
- 需要收集哪些报名信息

于是可生成表单向用户提问：

- `请输入活动名称` → `plain_text`
- `请选择报名截止日期` → `date_picker`
- `请选择页面风格` → `select_static`
- `请选择报名信息字段` → `multi_select_static`

### 用户需求

用户说：

> 明天下午三点提醒我开会。

### 大模型判断

信息已足够，不应生成表单追问。
