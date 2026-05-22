# 文本输入组件 `plain_text`

## 使用时机

当需要用户自由输入文本时使用。

例如：

- 请输入项目名称
- 请输入补充说明
- 请输入需求描述

## 注意

在提问类型中使用 `plain_text` 表示文本输入。

最终飞书卡片组件实际使用：

```json
"tag": "input"
```

## 组件写法

```json
{
  "tag": "input",
  "placeholder": {
    "tag": "plain_text",
    "content": "请输入"
  },
  "default_value": "",
  "width": "default",
  "name": "{{字段名}}",
  "margin": "0px 0px 0px 0px"
}
```

## 示例

```json
{
  "tag": "input",
  "placeholder": {
    "tag": "plain_text",
    "content": "请输入"
  },
  "default_value": "",
  "width": "default",
  "name": "project_name",
  "margin": "0px 0px 0px 0px"
}
```