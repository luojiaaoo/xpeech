# 日期选择组件 `date_picker`

## 使用时机

当只需要用户选择日期，不需要具体时间时使用。

例如：

- 请选择开始日期
- 请选择截止日期
- 请选择入职日期

## 组件写法

```json
{
  "tag": "date_picker",
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
  "tag": "date_picker",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择"
  },
  "width": "default",
  "name": "deadline_date",
  "margin": "0px 0px 0px 0px"
}
```