# 官方规范章节索引

本表按未修改的 starlark-spec.md 快照生成，忽略代码块内的注释标题。
原文共 4838 行，来源见 SKILL.md。

先按名称找到目标，再读取相邻行范围。下列终点到下一个标题之前；二级标题的范围只覆盖章节引言，其子章节在后续行中单列。

示例：调用 read_file 时，path 指向本技能目录下的 references/starlark-spec.md，offset 设为起始行号，limit 设为结束行号减起始行号加一。

| 原文章节 | 起始行 | 结束行 |
| --- | ---: | ---: |
| Overview | 24 | 53 |
| Contents | 54 | 208 |
| Lexical elements | 209 | 348 |
| ↳ String literals | 349 | 371 |
| ↳ ↳ String escapes | 372 | 495 |
| ↳ Bytes literals | 496 | 526 |
| ↳ Special tokens | 527 | 536 |
| Data types | 537 | 567 |
| ↳ None | 568 | 576 |
| ↳ Booleans | 577 | 612 |
| ↳ Integers | 613 | 653 |
| ↳ Floating-point numbers | 654 | 755 |
| ↳ Strings | 756 | 840 |
| ↳ Bytes | 841 | 881 |
| ↳ Lists | 882 | 938 |
| ↳ Tuples | 939 | 987 |
| ↳ Dictionaries | 988 | 1101 |
| ↳ Sets | 1102 | 1235 |
| ↳ Functions | 1236 | 1449 |
| ↳ Built-in functions | 1450 | 1466 |
| Name binding and variables | 1467 | 1642 |
| Value concepts | 1643 | 1657 |
| ↳ Identity and mutation | 1658 | 1712 |
| ↳ Freezing a value | 1713 | 1728 |
| ↳ Hashing | 1729 | 1756 |
| ↳ Collection types | 1757 | 1786 |
| ↳ Iteration | 1787 | 1810 |
| ↳ Indexing | 1811 | 1880 |
| Expressions | 1881 | 1921 |
| ↳ Identifiers | 1922 | 1931 |
| ↳ Literals | 1932 | 1942 |
| ↳ Parenthesized expressions | 1943 | 1988 |
| ↳ Dictionary expressions | 1989 | 2016 |
| ↳ List expressions | 2017 | 2036 |
| ↳ Unary operators | 2037 | 2091 |
| ↳ Binary operators | 2092 | 2126 |
| ↳ ↳ `or` and `and` | 2127 | 2167 |
| ↳ ↳ Comparisons | 2168 | 2212 |
| ↳ ↳ Arithmetic operations | 2213 | 2307 |
| ↳ ↳ Membership tests | 2308 | 2342 |
| ↳ ↳ String interpolation | 2343 | 2414 |
| ↳ Conditional expressions | 2415 | 2449 |
| ↳ Comprehensions | 2450 | 2517 |
| ↳ Function and method calls | 2518 | 2539 |
| ↳ Dot expressions | 2540 | 2582 |
| ↳ Subscript expressions | 2583 | 2636 |
| ↳ Slice expressions | 2637 | 2695 |
| ↳ Lambda expressions | 2696 | 2735 |
| Statements | 2736 | 2748 |
| ↳ Pass statements | 2749 | 2772 |
| ↳ Assignments | 2773 | 2813 |
| ↳ Augmented assignments | 2814 | 2850 |
| ↳ Function definitions | 2851 | 2960 |
| ↳ Return statements | 2961 | 2981 |
| ↳ Expression statements | 2982 | 2996 |
| ↳ If statements | 2997 | 3042 |
| ↳ For statements | 3043 | 3082 |
| ↳ Break and Continue | 3083 | 3109 |
| ↳ Load statements | 3110 | 3155 |
| Module execution | 3156 | 3195 |
| Built-in constants and functions | 3196 | 3214 |
| ↳ None | 3215 | 3218 |
| ↳ True and False | 3219 | 3222 |
| ↳ abs | 3223 | 3226 |
| ↳ any | 3227 | 3231 |
| ↳ all | 3232 | 3236 |
| ↳ bool | 3237 | 3241 |
| ↳ bytes | 3242 | 3264 |
| ↳ dict | 3265 | 3287 |
| ↳ dir | 3288 | 3306 |
| ↳ enumerate | 3307 | 3321 |
| ↳ fail | 3322 | 3346 |
| ↳ float | 3347 | 3367 |
| ↳ getattr | 3368 | 3382 |
| ↳ hasattr | 3383 | 3386 |
| ↳ hash | 3387 | 3403 |
| ↳ int | 3404 | 3444 |
| ↳ len | 3445 | 3451 |
| ↳ list | 3452 | 3459 |
| ↳ max | 3460 | 3476 |
| ↳ min | 3477 | 3493 |
| ↳ print | 3494 | 3511 |
| ↳ range | 3512 | 3557 |
| ↳ repr | 3558 | 3581 |
| ↳ reversed | 3582 | 3594 |
| ↳ set | 3595 | 3609 |
| ↳ sorted | 3610 | 3632 |
| ↳ str | 3633 | 3651 |
| ↳ tuple | 3652 | 3657 |
| ↳ type | 3658 | 3667 |
| ↳ zip | 3668 | 3682 |
| Built-in methods | 3683 | 3690 |
| ↳ bytes·elems | 3691 | 3703 |
| ↳ dict·clear | 3704 | 3715 |
| ↳ dict·get | 3716 | 3731 |
| ↳ dict·items | 3732 | 3742 |
| ↳ dict·keys | 3743 | 3753 |
| ↳ dict·pop | 3754 | 3771 |
| ↳ dict·popitem | 3772 | 3785 |
| ↳ dict·setdefault | 3786 | 3806 |
| ↳ dict·update | 3807 | 3836 |
| ↳ dict·values | 3837 | 3848 |
| ↳ list·append | 3849 | 3863 |
| ↳ list·clear | 3864 | 3875 |
| ↳ list·extend | 3876 | 3897 |
| ↳ list·index | 3898 | 3919 |
| ↳ list·insert | 3920 | 3938 |
| ↳ list·pop | 3939 | 3954 |
| ↳ list·remove | 3955 | 3969 |
| ↳ set·add | 3970 | 3984 |
| ↳ set·clear | 3985 | 3992 |
| ↳ set·difference | 3993 | 4013 |
| ↳ set·difference\_update | 4014 | 4036 |
| ↳ set·discard | 4037 | 4058 |
| ↳ set·intersection | 4059 | 4079 |
| ↳ set·intersection\_update | 4080 | 4102 |
| ↳ set·isdisjoint | 4103 | 4113 |
| ↳ set·issubset | 4114 | 4124 |
| ↳ set·issuperset | 4125 | 4135 |
| ↳ set·pop | 4136 | 4152 |
| ↳ set·remove | 4153 | 4171 |
| ↳ set·symmetric\_difference | 4172 | 4189 |
| ↳ set·symmetric\_difference\_update | 4190 | 4210 |
| ↳ set·union | 4211 | 4231 |
| ↳ set·update | 4232 | 4253 |
| ↳ string·capitalize | 4254 | 4263 |
| ↳ string·count | 4264 | 4276 |
| ↳ string·elems | 4277 | 4302 |
| ↳ string·endswith | 4303 | 4321 |
| ↳ string·find | 4322 | 4339 |
| ↳ string·format | 4340 | 4370 |
| ↳ string·index | 4371 | 4383 |
| ↳ string·isalnum | 4384 | 4394 |
| ↳ string·isalpha | 4395 | 4405 |
| ↳ string·isdigit | 4406 | 4416 |
| ↳ string·islower | 4417 | 4428 |
| ↳ string·isspace | 4429 | 4439 |
| ↳ string·istitle | 4440 | 4452 |
| ↳ string·isupper | 4453 | 4464 |
| ↳ string·join | 4465 | 4477 |
| ↳ string·lower | 4478 | 4486 |
| ↳ string·lstrip | 4487 | 4499 |
| ↳ string·partition | 4500 | 4513 |
| ↳ string·removeprefix | 4514 | 4528 |
| ↳ string·removesuffix | 4529 | 4543 |
| ↳ string·replace | 4544 | 4556 |
| ↳ string·rfind | 4557 | 4569 |
| ↳ string·rindex | 4570 | 4582 |
| ↳ string·rpartition | 4583 | 4591 |
| ↳ string·rsplit | 4592 | 4604 |
| ↳ string·rstrip | 4605 | 4617 |
| ↳ string·split | 4618 | 4646 |
| ↳ string·splitlines | 4647 | 4665 |
| ↳ string·startswith | 4666 | 4686 |
| ↳ string·strip | 4687 | 4700 |
| ↳ string·title | 4701 | 4711 |
| ↳ string·upper | 4712 | 4720 |
| Grammar reference | 4721 | 4838 |
