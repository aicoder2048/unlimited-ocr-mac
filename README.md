# Unlimited-OCR · Claude Code 技能

一个 **[Claude Code](https://claude.com/claude-code) 项目级技能**:在 Mac(Apple Silicon)上
**本地、离线**运行百度 [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 文档解析大模型,
把**图片、整个图片目录、多页扫描 PDF** 转成 Markdown(支持中英文、iPhone HEIC、版面识别、插图抽取)。

用法很简单:**在本项目里打开 Claude Code,直接对它说人话**——例如"把这个目录的扫描页 OCR 成
markdown"——技能会自动触发,在后台调用模型(CPU)完成,并把结果写到 `outputs/`。
你也可以直接敲命令行,见文末「CLI(进阶)」。

---

## 这是什么 / 为什么用 CPU

Unlimited-OCR 是 DeepSeek-V2 **MoE** 架构。它的专家路由在 PyTorch 的 **MPS(Metal / GPU)后端上数值
算错**——推理输出为空,而且 `PYTORCH_ENABLE_MPS_FALLBACK` 也救不了(算子是"实现了但算错",不触发回退)。
在 **CPU 上则完全正确**,这就是本技能采用的路径。官方模型代码里写死了 `.cuda()`,技能脚本**从不修改它**,
而是在运行时用 monkeypatch 把 CUDA 调用重定向到 CPU。

技能本体在 [`.claude/skills/unlimited-ocr/`](.claude/skills/unlimited-ocr/):一个 `SKILL.md`(告诉 Claude
何时触发、怎么用)+ 一个批处理脚本 `scripts/ocr.py`(模型只加载一次,循环处理所有页)。

---

## 安装

```bash
git clone https://github.com/aicoder2048/unlimited-ocr-mac.git
cd unlimited-ocr-mac
uv sync                                   # 建环境(torch / transformers / pymupdf / pillow-heif …)

# 下载模型权重(~6.3 GB,已 gitignore)到 models/
uv run huggingface-cli download baidu/Unlimited-OCR \
  --local-dir models/unlimited_ocr_official
```

装好后,**在这个项目目录里打开 Claude Code**,技能即自动可用(项目级技能,无需额外注册)。

---

## 在 Claude Code 里这样用(主要方式)

直接用自然语言对 Claude 说,它会读 `SKILL.md` 并在后台跑脚本。下面是几组真实对话演示。

### 演示 1 · 单张截图 / 收据

> **你**:识别一下 `~/Desktop/receipt.png` 上的金额和日期
>
> **Claude**:(触发 unlimited-ocr 技能,用 `--no-crop` 快速跑,约 6 秒)
> 识别结果:金额 ¥128.50,日期 2026-06-26。完整文本已存到 `outputs/receipt/result.md`。

### 演示 2 · iPhone HEIC 照片

> **你**:这张 iPhone 拍的 `~/Downloads/IMG_4421.HEIC` 帮我把字认出来
>
> **Claude**:(HEIC 自动转 PNG 并校正方向,再 OCR)识别完成,内容是…

### 演示 3 · 整个目录(扫描书)

> **你**:把 `inputs/某本书/` 这个目录里的扫描页全部 OCR 成 markdown
>
> **Claude**:(目录里图片按文件名自然排序逐张识别,合并成一个 `result.md`)
> 共 N 页,已写入 `outputs/某本书/result.md`,每页明细在 `pages/` 下。

### 演示 4 · 扫描 PDF

> **你**:OCR 这个扫描的 PDF:`~/scans/handbook.pdf`
>
> **Claude**:(逐页渲染再识别,默认高分辨率切片以保证密集小字)已生成合并 markdown…

### 演示 5 · 每页一个独立 md(逐页喂 AI)

> **你**:把这本扫描书 OCR 了,**每页存成一个单独的 md**,我要逐页喂给 AI 处理
>
> **Claude**:(加 `--split`)除合并的 `result.md` 外,还平铺出了 `page_0001.md`、`page_0002.md` …,
> 方便你一页一页地处理。

### 演示 6 · 大部头先取几页试水

> **你**:这本几百页的书别一次全跑,先 OCR 前 5 页让我看看质量
>
> **Claude**:(`--pages 1-5 --no-crop`)前 5 页已出,质量你看下;满意我再全量跑。

### 演示 7 · 速度优先 vs 质量优先

> **你**:这页是密密麻麻的小字目录,要识别得**尽量准**,慢点没关系
>
> **Claude**:(改用默认 **crop** 高分辨率切片模式,放弃 `--no-crop` 的提速)…
>
> 反过来,如果你说"**快点,普通照片够用就行**",Claude 会用 `--no-crop`(约 40× 提速)。

### 演示 8 · 真实案例:一本中文书

我们用它 OCR 过《稳重求胜·散户波段交易战法》的 32 张手机照片(封面+目录+第一章正文,12MP HEIC):

> **你**:把 `inputs/稳重求胜-散户波段交易战法-牛行健/` 整个目录 OCR 成 markdown,每页独立一份
>
> **Claude**:(32 页,`--no-crop --split`,约 15 分钟,增量落盘)完成,0 失败。
> 多数页准确率很高;有几页因**薄纸透印 / 照片带进对开页**产生了退化杂质(英文刷屏、整句重复),
> 我已自动清理,并提示你这些是拍摄问题。

> 💡 **实战经验**:薄纸书拍照时,**背面垫一张深色不透光的纸**可消除透印鬼影;**让单页摊平、别把对开页带进画面**,
> 能避免边缘图表把模型带崩。章首/整页图表页若仍不稳,可单独对它用默认 crop 重跑。

---

## 输出结构

无论怎么触发,结果都在 `outputs/<名字>/`:

```
outputs/<名字>/
  result.md            # 合并:全部页(整本一个文件,适合整体上下文)
  page_0001.md ...     # 每页一个独立 md(要求"每页独立"时,即 --split)
  pages/page_0001/
    result.md          # 该页 md(总是生成)
    result_with_boxes.jpg   # 带版面框的标注图
    images/0.jpg ...        # 模型抽出的插图(被 md 引用)
```

---

## CLI(进阶 / 可选)

技能背后就是一个脚本,你也可以**绕开 Claude 直接跑**。脚本自带 PEP 723 内联依赖,`uv run` 会自动装齐。

```bash
# 通用形式:<输入> 可以是单张图 / 图片目录 / PDF
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py <输入> [选项]

# 例:
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py receipt.heic --no-crop
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py ./scans/ --split
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py book.pdf --pages 1-20 --dpi 150
```

| 选项 | 作用 | 对应的自然语言意图 |
|---|---|---|
| `--no-crop` | 关闭高清切片,≈40× 提速(单图 ≈6s) | "快点 / 普通照片够用" |
| (默认 crop) | 高分辨率动态切片,密集小字更准(≈4–5 分钟/页 12MP) | "尽量准 / 密集小字" |
| `--split` | 额外平铺每页独立 md(`page_NNNN.md`) | "每页存一个单独的 md" |
| `--pages 1-5,8` | PDF 仅处理指定页(1 基) | "先看前几页 / 只要某几页" |
| `--dpi 180` | PDF 渲染分辨率(低=快,高=清) | — |
| `--out 名字` | 自定义输出子目录名 | — |
| `--prompt "..."` | OCR 提示词(默认 `document parsing.`;`Free OCR.`=纯文本) | — |

> `crop` / `base_size` / `image_size` 是 Unlimited-OCR **原生** `infer()` 参数(DeepSeek-OCR 血统的
> 动态高清切片),本项目只是把它们做成了 CLI 开关。

---

## 注意事项

- `models/`、`inputs/`、`outputs/` 均已 gitignore(大权重 / 个人扫描件 / 结果)。
- `models/` 是**纯净的厂商代码+权重**(技能从不修改),用上面那条 `huggingface-cli download` 随时可复现。
- **速度是 CPU 决定的**:几百页的书 + crop 可能要数小时;善用 `--no-crop` / 先取几页验证。
- `result.md` 增量落盘,长任务中断也保留已完成页。
- 模型对**图表占满、文字稀疏**的页面偶有重复/英文退化——多因拍摄(透印、对开页)而非读错真字;Claude 会自动清理这类杂质。
