---
name: ask-user-question
description: 用于向用户了解任务细节、收集信息、澄清需求、确认偏好或补全任务参数。支持组合生成单选、多选、输入框、日期选择、日期时间选择等表单控件。当你有问题需要问用户时触发。
always: true
internal: true
---

# 主动询问用户问题

当继续执行任务所需的信息缺失，且无法从上下文合理推断时，使用本 Skill 一次生成包含标题、副标题和问题列表的完整表单，并通过 `ask_user_question` 向用户追问。

## 调用参数

调用 `ask_user_question` 时，仅传入 `question` 参数，其值是完整表单对象的 JSON 字符串。将下面的对象序列化后传入：

```json
{
  "type": "form",
  "title": "需要补充信息",
  "subtitle": "请补充以下信息，方便继续处理任务",
  "submit_label": "提交",
  "fields": [
    {
      "type": "input",
      "name": "goal",
      "label": "希望达成什么目标？",
      "placeholder": "请简要描述你的目标"
    }
  ]
}
```

表单属性说明：

- `type`：固定为 `form`。
- `title`：表单标题，简短说明本次追问的目的，默认 `需要补充信息`。
- `subtitle`：副标题，用于补充背景或填写说明，可省略。系统自动在末尾追加填写时限。
- `submit_label`：提交按钮文案，默认 `提交`。
- `fields`：问题字段数组，建议一次 1 到 5 个问题。

标题、副标题和问题字段都放在同一个表单对象中，不再单独传入 `title`、`subtitle` 工具参数。系统在问题列表末尾自动追加供用户自定义填写的“补充说明”输入框，无需在 `fields` 中额外添加此类 field。

## 字段类型

### 输入框

用于收集自由文本。

```json
{
  "type": "input",
  "name": "project_goal",
  "label": "请描述希望达成的目标",
  "placeholder": "例如减少重复录入，提高处理效率",
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
  "name": "requested_sections",
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
5. `name` 使用英文小写加下划线，匹配 `^[a-z][a-z0-9_]*$`。
6. 选项应互斥、清晰、覆盖常见情况；不确定时可加入“其他”“不确定”“由模型推荐”等选项。
7. `question` 是完整表单对象的合法 JSON 字符串，包含 `type`、`title`、`subtitle`、`submit_label`、`fields`；一次生成后传入。
8. 无需额外生成用户自定义的“补充说明”“其他想法”等通用输入框 field，系统会自动追加。`fields` 只包含本次任务需要明确询问的具体问题。

## 标准示例

将以下完整表单对象序列化为 JSON 字符串，作为唯一的 `question` 参数传入：

```json
{
  "type": "form",
  "title": "请补充任务信息",
  "subtitle": "请说明任务类型、项目背景和期望完成日期",
  "submit_label": "提交",
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
      "name": "requested_sections",
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
      "name": "project_background",
      "label": "请描述项目背景与使用场景",
      "placeholder": "例如面向的用户和当前遇到的问题"
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
