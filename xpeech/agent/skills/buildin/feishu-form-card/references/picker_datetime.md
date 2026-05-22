# 日期时间选择组件 `picker_datetime`

## 使用时机

当需要用户选择具体日期和时间时使用。

例如：

- 请选择会议时间
- 请选择提醒时间
- 请选择任务截止时间

## 组件写法

```json
{
  "tag": "picker_datetime",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择"
  },
  "width": "default",
  "name": "{{字段名}}",
  "margin": "0px 0px 0px 0px"
}
```

## 示例

```json
{
  "tag": "picker_datetime",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择"
  },
  "width": "default",
  "name": "meeting_time",
  "margin": "0px 0px 0px 0px"
}
```