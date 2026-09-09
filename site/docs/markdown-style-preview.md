# Markdown Style Preview

这是一页独立的 Markdown 视觉样式册。页面保持普通 Markdown 写法，用同一组内容检查文字层级、强调、引用、代码和数据展示。

---

## 标题与正文

### 三级标题：反馈决定 agent 能走多远

高性能 kernel 的优化是一条很长的决策路径。程序越复杂，需要连续做出的结构选择越多。正文需要保持安静、清晰，让读者可以长时间阅读；标题负责建立节奏，而不与正文中的结论争夺注意力。

第二段用于观察段落间距。中英文混排包括 authored HIR、runtime twin、context length、CTA 和 shared memory，数字包括 2048、4.8 TB/s 与 32 × 128。

## 强调层级

普通正文承担叙述。**粗体表示作者希望读者记住的核心判断。** *斜体用于术语、语气和轻量强调。* ***粗斜体只留给一节中最重要的结论，不应频繁出现。***

同一段再观察混排：`analyze` 提供静态代价，`check` 验证 runtime twin；**反馈必须带有归因**，而不仅是一项最终耗时。

## 链接与行内代码

[TileFoundry](https://github.com/tile-ai/TileFoundry) 提供 authored HIR、runtime twin 和面向 agent 的命令行接口。运行 `tilefoundry analyze model.py:Model report.txt --memory` 可以得到流量与容量分析；变量如 `ctx_len`、`rmem` 和 `smem` 使用行内代码表示。

页内链接也使用同一套视觉语言，例如返回 [Markdown Style Preview](#markdown-style-preview)。

## 引用

> 要让 FA3 结构最终快过 FlashInfer，几个结构改动必须一起完成。孤立评估其中任何一个，测出来都可能没有收益；只有异步流水、warp 分工与寄存器预算同时成立，整体收益才会出现。

引用之后的正文应当自然回到主叙述，不需要额外的装饰或缩小字号。

## 列表

一次优化可以拆成三个问题：

1. 这份程序的性能差距来自哪里？
2. 下一项结构改动是否仍保持语义？
3. 当前 placement 给出了哪些新的优化线索？

也可以使用无序列表：

- `analyze` 给出工作量、流量、容量与 roofline；
- `check` 对每个输出应用显式判据；
- HIR 类型记录 shape、dtype、storage 和 mesh distribution。

## 代码块

```python
with Mesh(("cta",), layout=(8, 32), names=("row", "col")) as cta:
    with Mesh(("thread",), layout=(32, 8), names=("row", "col")) as thread:
        x_r = tf.reshard(
            x,
            (1, 8 @ cta.row, 16, 32 @ thread.row,
                32 @ cta.col, 8 @ thread.col, 8),
            "rmem",
        )
        y = tf.square(x_r)
```

```sh
tilefoundry analyze model.py:Model report.txt \
    --compute-cost --memory --roofline
```

## 表格

| context | 32 | 2048 | 16384 | 262080 |
| :--- | ---: | ---: | ---: | ---: |
| PyTorch reference | 34.67 μs | — | — | — |
| CUDA twin | **7.68 μs** | — | — | — |
| 距 roofline 下限 | 2.39× | 2.41× | 2.44× | 2.41× |

## 分隔线与脚注

分隔线只用于主题明显转换，不应出现在每个小节之间。

---

静态分析得到的是基于 target 规格和程序结构的性能界，不是真机测量结果。[^analysis]

[^analysis]: 真机性能仍然需要运行 runtime twin，并在固定设备、输入与测量方法下采集。
