---
name: ask-user-question
description: 用于向用户了解任务细节、收集信息、澄清需求、确认偏好或补全任务参数。支持组合生成下拉单选、下拉多选、输入框、日期选择、日期时间选择等表单控件。当你有问题需要问用户时触发。
always: true
---

# 主动询问用户问题

当你缺少继续执行任务所必需的信息，且无法从上下文合理推断时，使用本 Skill 生成通用问题表单 JSON，并通过 `ask_user_question` 向用户追问。

只在确实需要用户补充信息时提问。能通过上下文推断的内容，直接继续执行。

## 输出格式

调用 `ask_user_question` 时，`question` 参数必须是一个合法 JSON 对象字符串，结构如下：

```json
{
  "type": "form",
  "title": "需要补充信息",
  "submit_label": "提交",
  "include_customization": true,
  "fields": []
}
```

字段说明：

- `type`：固定为 `form`
- `title`：表单标题，简短说明本次追问的目的
- `submit_label`：提交按钮文案，默认 `提交`
- `include_customization`：是否追加一个 `user_customization` 自由输入框，默认 `true`
- `fields`：问题字段数组，建议一次 1 到 5 个

## 字段类型

### 输入框

用于收集自由文本。

```json
{
  "type": "input",
  "name": "additional_context",
  "label": "请补充具体需求或背景信息",
  "placeholder": "请输入补充说明",
  "default_value": ""
}
```

### 单选

用于让用户从多个互斥选项中选择一个。

```json
{
  "type": "select",
  "name": "task_type",
  "label": "请选择任务类型",
  "placeholder": "请选择",
  "options": [
    { "label": "需求梳理", "value": "requirement_analysis" },
    { "label": "方案设计", "value": "solution_design" },
    { "label": "内容生成", "value": "content_generation" },
    { "label": "其他", "value": "other" }
  ]
}
```

### 多选

用于让用户选择多个选项。

```json
{
  "type": "multi_select",
  "name": "required_sections",
  "label": "请选择需要包含的内容",
  "placeholder": "请选择，可多选",
  "options": [
    { "label": "背景分析", "value": "background_analysis" },
    { "label": "执行步骤", "value": "execution_steps" },
    { "label": "风险提示", "value": "risk_notes" },
    { "label": "结果示例", "value": "examples" }
  ]
}
```

### 日期

用于收集某一天。

```json
{
  "type": "date",
  "name": "deadline_date",
  "label": "请选择期望完成日期",
  "placeholder": "请选择日期"
}
```

### 日期时间

用于收集精确到时间点的信息。

```json
{
  "type": "datetime",
  "name": "meeting_datetime",
  "label": "请选择会议开始时间",
  "placeholder": "请选择日期和时间"
}
```

## 生成规则

1. 只询问完成任务所必需的信息。
2. 优先使用 `select` 或 `multi_select` 降低用户输入成本。
3. 对开放性、个性化、背景类问题使用 `input`。
4. 对日期使用 `date`，对具体时间点使用 `datetime`。
5. `name` 必须唯一、稳定、语义化，使用英文小写加下划线。
6. `name` 必须匹配 `^[a-z][a-z0-9_]*$`。
7. 不要生成 `required` 字段；所有主动询问的问题默认都是必填。
8. 当 `include_customization` 为 `true` 时，不要使用保留字段名 `user_customization`。
9. 选项应互斥、清晰、覆盖常见情况；不确定时可加入“其他”“不确定”“由模型推荐”等选项。
10. `question` 必须是合法 JSON 对象字符串，不要在 JSON 外附加解释。

## 标准示例

```json
{
  "type": "form",
  "title": "请补充任务信息",
  "submit_label": "提交",
  "include_customization": true,
  "fields": [
    {
      "type": "select",
      "name": "task_type",
      "label": "请选择你希望处理的任务类型",
      "placeholder": "请选择",
      "options": [
        { "label": "需求梳理", "value": "requirement_analysis" },
        { "label": "方案设计", "value": "solution_design" },
        { "label": "内容生成", "value": "content_generation" },
        { "label": "其他", "value": "other" }
      ]
    },
    {
      "type": "multi_select",
      "name": "required_sections",
      "label": "请选择需要包含的内容",
      "placeholder": "请选择，可多选",
      "options": [
        { "label": "背景分析", "value": "background_analysis" },
        { "label": "执行步骤", "value": "execution_steps" },
        { "label": "风险提示", "value": "risk_notes" },
        { "label": "结果示例", "value": "examples" }
      ]
    },
    {
      "type": "input",
      "name": "additional_context",
      "label": "请补充具体需求或背景信息",
      "placeholder": "请输入补充说明"
    },
    {
      "type": "date",
      "name": "deadline_date",
      "label": "请选择期望完成日期",
      "placeholder": "请选择日期"
    }
  ]
}
```
