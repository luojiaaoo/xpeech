---
name: rss-reader
short_description: RSS 订阅与资讯
description: 按主题查找内建 RSS/Atom 订阅源，读取最新文章和技术动态并生成带来源链接的摘要；用于 RSS 源推荐、AI/科技简报、技术资讯汇总和订阅清单导出。
---

# RSS 订阅与资讯阅读

## 选择订阅源

目录包含 9 类、43 个 RSS/Atom 地址。

- 用户指定源或地址时，优先使用用户指定的范围；目录不是白名单。
- 用户仅要订阅地址时，直接返回名称、分类和地址，无需抓取文章。
- 生成简报时，先选择与主题相关的少量源；只有用户要求全量巡检或汇总时才遍历整个目录。
- HN 和 Product Hunt 等聚合源用于发现线索，重要事实尽量追溯原文。

## 抓取与解析

使用 `web_fetch` 读取订阅地址。

| 格式 | 条目 | 标题 | 原文链接 | 时间 | 摘要 |
| --- | --- | --- | --- | --- | --- |
| RSS 2.0 | `channel/item` | `title` | `link` | `pubDate`，兼顾 `dc:date` | `description` / `content:encoded` |
| Atom | `feed/entry` | `title` | `link[@rel='alternate']/@href`，其次无 `rel` 的链接 | `published`，缺失时用 `updated` | `summary` / `content` |

Atom 默认命名空间为 `http://www.w3.org/2005/Atom`，解析时显式绑定；其他 XML 命名空间按文档声明处理。相对链接按文档的 `xml:base` 和订阅 URL 解析。需要文章细节时，再用 `web_fetch` 读取原文；仅有标题或摘要时明确说明依据的范围。

## 筛选与汇总

- 按用户指定的主题、时间范围和条数筛选。未指定时间时读取近期条目，并在输出中说明实际覆盖的时间范围。
- RSS 日期与 Atom 日期格式不同，解析后统一到用户时区再筛选。缺失时间的条目标记“时间未知”，不要将抓取时间当作发布时间；按 `updated` 筛选时注明是更新时间。
- 同一原文 URL 或相同事件合并展示，保留相关来源；不要随意删除影响文章身份的 URL 查询参数。
- 默认用中文概括，保留产品名、版本号和论文标题等关键标识。每条提供标题、来源、发布时间（若有）、简要内容和可点击的原文链接。
- 简报开头说明主题、时间范围和读取的源；末尾仅在存在失败或覆盖不足时说明限制。无匹配条目时直接说明，不用过期条目填充。

## 可用性与导出

目录中的检测日期和可用率来自用户提供的历史清单，不代表当前网络可达性。检查源时，需同时确认 HTTP 请求成功、响应是可解析的 RSS/Atom，并区分“有效但暂无条目”和“请求或解析失败”；返回 200 的 HTML 页面不算有效订阅。

部分源可能因网络、代理、限流或站点变更失败。说明实际错误，继续处理其他相关源；一次失败不能证明订阅永久失效，也不能直接归因为需要代理。

用户要求导出订阅清单时，可按分类生成 Markdown 或 OPML。OPML 使用 `outline` 分组，订阅节点包含 `text`、`title`、`type="rss"`、`xmlUrl`，通过 XML 序列化器转义 URL 中的 `&` 等字符。保存到用户工作区；内建目录是只读资源。

# RSS 源目录

## 分类索引

| 分类 | 源数量 |
| --- | ---: |
| AI | 7 |
| 中文 / 国内 | 6 |
| 技术媒体 | 6 |
| Hacker News | 5 |
| Big Tech / 工程博客 | 9 |
| 安全 | 3 |
| 周刊 / Newsletter | 4 |
| 设计 / 产品 | 2 |
| 学术 | 1 |
| **合计** | **43** |

## AI

| 名称 | 订阅地址 |
| --- | --- |
| OpenAI Blog | `https://openai.com/news/rss.xml` |
| AIHOT 每日简报 | `https://aihot.news/feed/daily.xml` |
| arXiv AI (cs.AI) | `https://rss.arxiv.org/rss/cs.AI` |
| arXiv CV (cs.CV) | `https://rss.arxiv.org/rss/cs.CV` |
| Hacker News AI | `https://hnrss.org/newest?q=AI` |
| Hacker News LLM | `https://hnrss.org/newest?q=LLM` |
| Simon Willison | `https://simonwillison.net/atom/everything/` |

## 中文 / 国内

| 名称 | 订阅地址 |
| --- | --- |
| 少数派 SSPAI | `https://sspai.com/feed` |
| IT之家 | `https://www.ithome.com/rss/` |
| FreeBuf | `https://www.freebuf.com/feed` |
| 安全客 AnQuanKe | `https://api.anquanke.com/data/v1/rss` |
| 阮一峰周刊 | `https://www.ruanyifeng.com/blog/atom.xml` |
| 纳西论坛 | `https://forum.naixi.net/forum.php?mod=rss` |

## 技术媒体

| 名称 | 订阅地址 |
| --- | --- |
| TechCrunch | `https://techcrunch.com/feed/` |
| Ars Technica | `https://feeds.arstechnica.com/arstechnica/index` |
| MIT Technology Review | `https://www.technologyreview.com/feed/` |
| Wired | `https://www.wired.com/feed/rss` |
| Joho no Todai (EN) | `https://joho-todai.com/en/rss/` |
| Joho no Todai (JP) | `https://joho-todai.com/pinterest-rss/` |

## Hacker News

| 名称 | 订阅地址 |
| --- | --- |
| HN Frontpage | `https://hnrss.org/frontpage` |
| HN Newest | `https://hnrss.org/newest` |
| HN Best | `https://hnrss.org/best` |
| HN Ask | `https://hnrss.org/ask` |
| HN Show | `https://hnrss.org/show` |

## Big Tech / 工程博客

| 名称 | 订阅地址 |
| --- | --- |
| Netflix Tech Blog | `https://netflixtechblog.com/feed` |
| AWS | `https://aws.amazon.com/blogs/aws/feed/` |
| Cloudflare | `https://blog.cloudflare.com/rss/` |
| Vercel | `https://vercel.com/atom` |
| Supabase | `https://supabase.com/rss.xml` |
| Stripe | `https://stripe.com/blog/feed.rss` |
| Spotify Engineering | `https://engineering.atspotify.com/feed/` |
| Meta Engineering | `https://engineering.fb.com/feed/` |
| Mozilla Hacks | `https://hacks.mozilla.org/feed/` |

## 安全

| 名称 | 订阅地址 |
| --- | --- |
| Krebs on Security | `https://krebsonsecurity.com/feed/` |
| Schneier on Security | `https://www.schneier.com/feed/` |
| CISA News | `https://www.cisa.gov/news.xml` |

## 周刊 / Newsletter

| 名称 | 订阅地址 |
| --- | --- |
| JavaScript Weekly | `https://javascriptweekly.com/rss/` |
| This Week in Rust | `https://this-week-in-rust.org/atom.xml` |
| Golang Weekly | `https://golangweekly.com/rss/` |
| ByteByteGo | `https://blog.bytebytego.com/feed` |

## 设计 / 产品

| 名称 | 订阅地址 |
| --- | --- |
| Dribbble Stories | `https://dribbble.com/stories.rss` |
| Product Hunt | `https://www.producthunt.com/feed` |

## 学术

| 名称 | 订阅地址 |
| --- | --- |
| Nature | `https://www.nature.com/nature.rss` |
