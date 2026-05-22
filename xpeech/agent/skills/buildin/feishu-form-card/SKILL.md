---
name: feishu-form-question-card
description: 生成飞书卡片 2.0 JSON 数组。支持组合生成下拉单选、下拉多选、输入框、日期选择、日期时间选择等表单控件。当需要主动向用户收集信息、澄清需求、确认偏好或补全任务参数时使用。
---

# 主动询问用户问题

## 适用场景

当你需要在继续执行任务前，主动向用户询问一个或多个问题时，使用本 Skill 生成交互式表单，通过 joyride_request_human_input 向用户追问关键信息。

适用场景包括但不限于：

- 需求不完整，需要用户补充关键信息
- 存在多个可选方案，需要用户选择
- 需要确认时间、日期、范围、偏好或约束条件
- 需要一次性收集多个字段，避免多轮追问
- 需要结构化收集用户输入，便于后续自动处理

## 表单能力

本 Skill 支持以下 5 种表单类型，可自由组合使用：

1. 下拉单选：`select_static`
2. 下拉多选：`multi_select_static`
3. 输入框：`input`
4. 日期选择：`date_picker`
5. 日期时间选择：`picker_datetime`

## 输出要求

调用本 Skill 时，应输出一个 JSON 数组。

数组中的每个问题通常由两部分组成：

1. 一个标题说明组件：`div`
2. 一个表单输入组件：如 `select_static`、`multi_select_static`、`input`、`date_picker`、`picker_datetime`

每个表单组件必须包含唯一的 `name` 字段，用于后续识别用户答案。

## 通用字段规范

### 标题组件

每个问题前建议使用 `div` 作为标题说明。

```json
{
  "tag": "div",
  "text": {
    "tag": "plain_text",
    "content": "问题标题",
    "text_size": "normal_v2",
    "text_align": "left",
    "text_color": "default"
  },
  "margin": "0px 0px 0px 0px"
}
```

### 表单组件通用字段

表单组件建议包含以下字段：

- `tag`：组件类型
- `placeholder`：占位提示
- `width`：建议使用 `fill`
- `name`：唯一字段名
- `margin`：建议使用 `0px 0px 0px 0px`

字段名 `name` 应具备语义，例如：

- `project_type`
- `target_date`
- `user_requirements`
- `preferred_options`
- `meeting_time`

避免使用无意义或重复的字段名。

## 组件模板

### 1. 下拉单选

用于让用户从多个选项中选择一个答案。

适合场景：

- 选择任务类型
- 选择优先级
- 选择风格
- 选择是否继续
- 选择单一方案

```json
{
  "tag": "select_static",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择"
  },
  "options": [
    {
      "text": {
        "tag": "plain_text",
        "content": "选项1"
      },
      "value": "option_1",
      "icon": {
        "tag": "standard_icon",
        "token": "signature_outlined"
      }
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "选项2"
      },
      "value": "option_2",
      "icon": {
        "tag": "standard_icon",
        "token": "signature_outlined"
      }
    }
  ],
  "type": "default",
  "width": "fill",
  "name": "single_choice",
  "margin": "0px 0px 0px 0px"
}
```

### 2. 下拉多选

用于让用户从多个选项中选择多个答案。

适合场景：

- 选择多个需求
- 选择多个功能模块
- 选择多个偏好
- 选择多个限制条件

```json
{
  "tag": "multi_select_static",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择，可多选"
  },
  "options": [
    {
      "text": {
        "tag": "plain_text",
        "content": "选项1"
      },
      "value": "option_1",
      "icon": {
        "tag": "standard_icon",
        "token": "signature_outlined"
      }
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "选项2"
      },
      "value": "option_2",
      "icon": {
        "tag": "standard_icon",
        "token": "signature_outlined"
      }
    }
  ],
  "type": "default",
  "width": "fill",
  "name": "multiple_choices",
  "margin": "0px 0px 0px 0px"
}
```

### 3. 输入框

用于收集用户的自由文本输入。

适合场景：

- 描述需求
- 补充背景
- 输入名称
- 输入备注
- 输入自定义内容

```json
{
  "tag": "input",
  "placeholder": {
    "tag": "plain_text",
    "content": "请输入"
  },
  "default_value": "",
  "width": "fill",
  "label": {
    "tag": "plain_text",
    "content": ""
  },
  "label_position": "top",
  "name": "text_input",
  "margin": "0px 0px 0px 0px"
}
```

### 4. 日期选择

用于收集某一天的日期。

适合场景：

- 截止日期
- 开始日期
- 目标日期
- 生日
- 活动日期

```json
{
  "tag": "date_picker",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择日期"
  },
  "width": "fill",
  "name": "selected_date",
  "margin": "0px 0px 0px 0px"
}
```

### 5. 日期时间选择

用于收集精确到时间的日期时间信息。

适合场景：

- 会议时间
- 提醒时间
- 发布时间
- 预约时间
- 任务执行时间

```json
{
  "tag": "picker_datetime",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择日期和时间"
  },
  "width": "fill",
  "name": "selected_datetime",
  "margin": "0px 0px 0px 0px"
}
```

## 使用原则

### 何时提问

只有在缺少必要信息，且无法合理推断时，才主动向用户提问。

如果问题可以通过上下文推断，优先直接继续执行，不要过度打断用户。

### 问题数量

一次表单中建议包含 1 到 5 个问题。

如果问题过多，应优先询问最关键的信息，避免给用户造成负担。

### 问题设计

问题应清晰、具体、可回答。

推荐：

- “请选择你希望生成的文档类型”
- “请选择需要包含的功能模块”
- “请输入项目背景或补充说明”
- “请选择期望完成日期”
- “请选择会议开始时间”

避免：

- “你想要什么？”
- “还有吗？”
- “请补充一下”
- “选择一下”

### 选项设计

下拉选项应互斥、清晰、覆盖常见情况。

如果存在不确定情况，可以加入：

- “不确定”
- “其他”
- “暂不决定”
- “由模型推荐”

### 字段命名

`name` 字段必须稳定、唯一、语义化。

推荐使用英文小写加下划线：

```text
task_type
priority_level
required_features
additional_context
deadline_date
meeting_datetime
```

不推荐：

```text
Select_3w2jckvrbtg
Input_yuvaqw12aek
field1
aaa
test
```

## 标准输出示例

以下示例用于在开始生成方案前，主动询问用户任务类型、需要的功能、补充说明、截止日期和会议时间。

```json
[
  {
    "tag": "div",
    "text": {
      "tag": "plain_text",
      "content": "请选择你希望处理的任务类型",
      "text_size": "normal_v2",
      "text_align": "left",
      "text_color": "default"
    },
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "select_static",
    "placeholder": {
      "tag": "plain_text",
      "content": "请选择"
    },
    "options": [
      {
        "text": {
          "tag": "plain_text",
          "content": "需求梳理"
        },
        "value": "requirement_analysis",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "方案设计"
        },
        "value": "solution_design",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "内容生成"
        },
        "value": "content_generation",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "其他"
        },
        "value": "other",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      }
    ],
    "type": "default",
    "width": "fill",
    "name": "task_type",
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "div",
    "text": {
      "tag": "plain_text",
      "content": "请选择需要包含的功能或内容",
      "text_size": "normal_v2",
      "text_align": "left",
      "text_color": "default"
    },
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "multi_select_static",
    "placeholder": {
      "tag": "plain_text",
      "content": "请选择，可多选"
    },
    "options": [
      {
        "text": {
          "tag": "plain_text",
          "content": "背景分析"
        },
        "value": "background_analysis",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "执行步骤"
        },
        "value": "execution_steps",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "风险提示"
        },
        "value": "risk_notes",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      },
      {
        "text": {
          "tag": "plain_text",
          "content": "结果示例"
        },
        "value": "examples",
        "icon": {
          "tag": "standard_icon",
          "token": "signature_outlined"
        }
      }
    ],
    "type": "default",
    "width": "fill",
    "name": "required_sections",
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "div",
    "text": {
      "tag": "plain_text",
      "content": "请补充你的具体需求或背景信息",
      "text_size": "normal_v2",
      "text_align": "left",
      "text_color": "default"
    },
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "input",
    "placeholder": {
      "tag": "plain_text",
      "content": "请输入补充说明"
    },
    "default_value": "",
    "width": "fill",
    "label": {
      "tag": "plain_text",
      "content": ""
    },
    "label_position": "top",
    "name": "additional_context",
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "div",
    "text": {
      "tag": "plain_text",
      "content": "请选择期望完成日期",
      "text_size": "normal_v2",
      "text_align": "left",
      "text_color": "default"
    },
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "date_picker",
    "placeholder": {
      "tag": "plain_text",
      "content": "请选择日期"
    },
    "width": "fill",
    "name": "deadline_date",
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "div",
    "text": {
      "tag": "plain_text",
      "content": "如需安排沟通，请选择会议时间",
      "text_size": "normal_v2",
      "text_align": "left",
      "text_color": "default"
    },
    "margin": "0px 0px 0px 0px"
  },
  {
    "tag": "picker_datetime",
    "placeholder": {
      "tag": "plain_text",
      "content": "请选择日期和时间"
    },
    "width": "fill",
    "name": "meeting_datetime",
    "margin": "0px 0px 0px 0px"
  }
]
```

## 生成规则

当你根据用户任务生成表单时，请遵循以下规则：

1. 先判断当前任务缺少哪些关键信息。
2. 只询问完成任务所必需的信息。
3. 优先使用结构化选项，减少用户输入成本。
4. 对开放性、个性化或背景类问题使用输入框。
5. 对日期类信息使用 `date_picker`。
6. 对具体时间点使用 `picker_datetime`。
7. 每个问题标题必须简洁明确。
8. 每个控件的 `name` 必须唯一。
9. 输出必须是合法 JSON 数组。
10. 不要在 JSON 外输出额外解释文本。

## 输出格式约束

最终输出必须满足：

- 顶层结构是 JSON 数组
- 数组元素只能是表单相关组件对象
- 不要输出 Markdown
- 不要输出解释说明
- 不要输出代码块标记
- 不要输出 JSON 之外的任何文本
- 所有字符串必须使用双引号
- 所有 `name` 字段必须唯一且语义化

