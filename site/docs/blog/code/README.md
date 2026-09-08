# hello world 的源码与原始输出

[十八行的 hello world](../AI-compilers-in-the-agentic-era.md#hello) 那一节的三个数字来自这里。

## 环境

| | |
| --- | --- |
| 镜像 | `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-foundry-dev` |
| 源码 | TileFoundry `2306bf0`，挂进容器，`PYTHONPATH` 指向 `src` |
| 机器 | 一张 `NVIDIA H200`（`analyze` 只读 target 声明，不上卡） |

镜像里的 `/usr/bin/tilefoundry` 是一行 `exec python3 -m tilefoundry.cli "$@"`，包本身没有安装，所以 `PYTHONPATH` 必须给。

```sh
docker run --rm --gpus all \
  -v <TileFoundry checkout>:/work -v "$PWD:/out" -w /out \
  -e PYTHONPATH=/work/src \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-foundry-dev \
  tilefoundry analyze /out/<SOURCE>:<Module>.chain /out/<REPORT> \
              --compute-cost --memory --roofline
```

## 三次运行

| 源码 | 原始输出 | 结果 |
| --- | --- | --- |
| `hello_unplaced.py` | `raw_hello_unplaced.txt` | `ideal-ns=984 bound-by=memory` |
| `hello_placed.py` | `raw_hello_placed.txt` | `ideal-ns=219 bound-by=memory` |
| `hello_placed_toobig.py` | `raw_hello_refuse.txt` | 拒绝：`rmem` 要 6291456 B，target 声明 262144 B |

前两份的差别只有放置：`hello_placed.py` 多一个 `Mesh` 和两次 `reshard`，算术相同。第三份与第二份的差别只有 `T, D` 从 `1024, 256` 改成 `4096, 2048`。

`analyze` 的报告写进它的位置参数指向的文件，标准输出不带报告的任何一部分；拒绝走标准错误，所以 `raw_hello_refuse.txt` 收的是 stderr。

`analyze` 报的是算出来的界，同一程序重复运行给出同一个数，所以这里每份原始输出只留一次。
