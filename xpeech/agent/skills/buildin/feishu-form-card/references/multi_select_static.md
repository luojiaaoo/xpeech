# 多选组件 `multi_select_static`

## 使用时机

当用户可以同时选择多个答案时使用。

例如：

- 请选择需要支持的功能
- 请选择发布渠道
- 请选择需要包含的模块

## 组件写法

```json
{
  "tag": "multi_select_static",
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
  "name": "{{字段名}}",
  "margin": "0px 0px 0px 0px"
}
```

## 示例

```json
{
  "tag": "multi_select_static",
  "placeholder": {
    "tag": "plain_text",
    "content": "请选择"
  },
  "options": [
    {
      "text": {
        "tag": "plain_text",
        "content": "登录注册"
      },
      "value": "auth"
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "消息通知"
      },
      "value": "notification"
    },
    {
      "text": {
        "tag": "plain_text",
        "content": "数据看板"
      },
      "value": "dashboard"
    }
  ],
  "type": "default",
  "width": "default",
  "name": "required_features",
  "margin": "0px 0px 0px 0px"
}
```