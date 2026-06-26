# Unlimited-OCR on Apple Silicon（苹果芯片本地 OCR）

在 Mac（Apple Silicon）上**本地、离线**运行百度
[Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 文档解析大模型,
封装成一个可复用的 Claude Code 技能,把**单张图片、整个图片目录、多页扫描 PDF**
一键转成 Markdown(支持中英文、iPhone HEIC、版面识别、插图抽取)。

> 适合把纸质书/扫描件/手机拍的文档批量 OCR 成 md,再交给 AI 做下游处理(总结、问答、翻译、结构化)。

---

## 为什么用 CPU 而不是 GPU(MPS)

Unlimited-OCR 是 DeepSeek-V2 **MoE** 架构。它的专家路由在 PyTorch 的 **MPS(Metal)后端上数值算错**
——推理输出为空,而且 `PYTORCH_ENABLE_MPS_FALLBACK` 也救不了(算子是"实现了但算错",不触发回退)。
在 **CPU 上则完全正确**,这就是本项目采用的路径。官方模型代码里写死了 `.cuda()`,我们**从不修改它**——
脚本在运行时用 monkeypatch 把 CUDA 调用重定向到 CPU。

M-series + 大内存上,CPU 推理速度可接受(MoE 每次只激活 6/64 专家)。社区的 MLX 转换版当时还是半成品(SAM
相对位置编码未实现、推理直接崩),所以没有采用。

---

## 安装

```bash
uv sync                                   # 建环境(torch / transformers / pymupdf / pillow-heif …)

# 下载模型权重(~6.3 GB,已 gitignore)到 models/
uv run huggingface-cli download baidu/Unlimited-OCR \
  --local-dir models/unlimited_ocr_official
```

---

## 快速开始

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py <输入> [选项]
```

`<输入>` 可以是:**单张图片** / **图片目录** / **PDF 文件**。结果默认写到 `outputs/<输入名>/`。

---

## 用法演示

### Demo 1 · 单张截图 / 收据(最快)

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py receipt.png --no-crop
```
普通照片、截图、收据用 `--no-crop`:整页缩放跑一次,**约 6 秒/张**。

### Demo 2 · iPhone HEIC 照片

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py IMG_4421.HEIC --no-crop
```
HEIC 自动转 PNG(并应用 EXIF 方向),无需手动转码。

### Demo 3 · 整个图片目录

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py ./scans/
```
目录里的图片按文件名**自然排序**(page1, page2, …, page10)逐张 OCR,合并成一个 `result.md`。

### Demo 4 · 扫描的整本书(PDF)

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py book.pdf
```
PDF 用 pymupdf 逐页渲染再 OCR。默认开启高分辨率切片(crop),适合密集小字。

### Demo 5 · 大书只先取部分页试水

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py book.pdf --pages 1-5 --no-crop
```
几百页的书别一上来全跑。先 `--pages 1-5 --no-crop` 看质量,满意再全量。

### Demo 6 · 每页输出一个独立 md(逐页喂 AI)

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py book.pdf --split
```
除合并的 `result.md` 外,额外平铺出 `page_0001.md`、`page_0002.md` …,方便逐页处理。

### Demo 7 · 速度优先 vs 质量优先

| 模式 | 命令 | 速度(实测) | 适用 |
|---|---|---|---|
| 速度优先 | `--no-crop` | 单图 ≈6s;8 张 12MP ≈3 分钟 | 普通照片、截图、幻灯片、大字 |
| 质量优先 | (默认 crop) | ≈4–5 分钟/页(12MP) | 密集小字、扫描书、表格、需要每个字 |

> `crop` 是 Unlimited-OCR **原生**的动态高清切片(DeepSeek-OCR 血统),本项目只是把它做成了 `--no-crop` 开关。

### Demo 8 · 实战:中文书目录 OCR

输入 8 张 iPhone 拍的《稳重求胜·散户波段交易战法》封面+目录(12MP HEIC),`--no-crop` 约 3 分钟:

```bash
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py ./book_photos/ --no-crop --split
```

输出片段(中文目录,含章节层级与页码,准确率 95%+):

```markdown
| 目录 |
第一章 只做上升段：波段交易的核心
第一节 波段操作与波段级别 / 2
一、波段交易的时间框架 / 3
二、波段交易模式——大波段 / 6
第二节 波段交易获利的三个前提 / 12
...
```

---

## 输出结构

```
outputs/<名>/
  result.md            # 合并:全部页(整本一个文件,适合整体上下文)
  page_0001.md ...     # 每页一个独立 md(加 --split 时)
  pages/page_0001/
    result.md          # 该页 md(总是生成)
    result_with_boxes.jpg   # 带版面框的标注图
    images/0.jpg ...        # 模型抽出的插图(被 md 引用)
```

---

## 选项速查

| 选项 | 作用 |
|---|---|
| `--no-crop` | 关闭高清切片,≈40× 提速;普通图够用 |
| `--split` | 额外平铺每页独立 md(`page_NNNN.md`) |
| `--out 名字` | 自定义输出子目录名(默认取输入名) |
| `--pages 1-5,8` | PDF 仅处理指定页(1 基) |
| `--dpi 180` | PDF 渲染分辨率(低=快,高=清) |
| `--prompt "..."` | OCR 提示词(默认 `document parsing.`;可用 `Free OCR.` 纯文本) |
| `--model-dir` / `--out-root` | 也可用环境变量 `UNLIMITED_OCR_MODEL_DIR` / `UNLIMITED_OCR_OUT` |

脚本自带 PEP 723 内联依赖,`uv run` 会自动装齐依赖(可脱离项目 pyproject 单文件运行)。

---

## 在 Claude Code 里直接用

这是个项目级技能(`.claude/skills/unlimited-ocr/`)。在本项目里对 Claude 说一句即可自动触发,例如:

> "把 `inputs/书名/` 这一目录的扫描页 OCR 成 markdown"
> "识别这张 HEIC 收据上的金额"
> "read this scanned PDF into markdown"

---

## 注意事项

- `models/`、`inputs/`、`outputs/` 均已 gitignore(大权重 / 个人扫描件 / 结果)。
- `models/` 是**纯净的厂商代码+权重**(我们从不修改),用上面那条 `huggingface-cli download` 随时可复现。
- CPU 推理较慢:几百页的书 + crop 可能要数小时;善用 `--no-crop` / `--pages` 先验证再全量。
- `result.md` 增量落盘,长任务中断也保留已完成页。
