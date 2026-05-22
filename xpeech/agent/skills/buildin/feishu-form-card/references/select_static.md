# 单选组件 `select_static`

## 使用时机

当用户只能从多个答案中选择一个时使用。

例如：

- 请选择输出格式
- 请选择目标平台
- 请选择优先级

## 组件写法

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
        "content": "{{选项文案1}}"
      },
      "value": "{{选项值1}}"
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "{{选项文案2}}"
      },
      "value": "{{选项值2}}"
    }
  ],
  "type": "default",
  "width": "default",
  "name": "{{字段名}}"
}
```

## 示例

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
        "content": "Markdown"
      },
      "value": "markdown"
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "JSON"
      },
      "value": "json"
    }
  ],
  "type": "default",
  "width": "default",
  "name": "output_format"
}
```