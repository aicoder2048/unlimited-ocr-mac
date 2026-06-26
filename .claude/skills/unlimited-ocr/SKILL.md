---
name: unlimited-ocr
description: >-
  Local OCR / document parsing for visual files on disk, using the local
  baidu/Unlimited-OCR model on Apple Silicon (CPU). Invoke whenever the user points at a
  local path and wants the text or layout pulled out into saved Markdown: a single image
  (screenshot, photo, receipt, menu, invoice, iPhone HEIC), a whole folder of page images,
  or a multi-page scanned PDF (a scanned book or document). Trigger phrasings: "OCR this",
  "extract the text from this scan / PDF / photo", "read this scanned book", "turn this
  folder of scans into markdown", "识别这张图/扫描件/发票/古籍上的文字", "把扫描的 pdf 转成文字".
  Handles JPG/PNG/WebP/TIFF/BMP, iPhone HEIC, image directories, and multi-page scanned
  PDFs (English + Chinese), writing results under outputs/<name>/. Especially preferred
  over reading images one-by-one when there are many pages/files or the user wants a saved
  markdown deliverable — even if they don't say "Unlimited-OCR" by name. Do NOT use for:
  editable/text-based PDFs, Office/Word/Excel files, audio or video, image
  compression/resizing, or translating text that has already been extracted.
---

# Unlimited-OCR (local, Apple Silicon)

Wraps the locally-downloaded **baidu/Unlimited-OCR** model to turn images / PDFs into
Markdown with layout grounding. One bundled script, `scripts/ocr.py`, handles all input
types and writes results under `outputs/<name>/`.

## Why this exists / key facts

- The model is a **DeepSeek-V2 MoE**. Its expert routing is **numerically wrong on the
  PyTorch MPS (Metal) backend** — inference returns empty output. So the script runs on
  **CPU**, which is the only verified-correct path. (`PYTORCH_ENABLE_MPS_FALLBACK` does
  not help: the ops are implemented-but-wrong, so they don't fall back.)
- The official model code hardcodes `.cuda()`. The script **does not modify any official
  file** — it monkeypatches `torch` at startup to redirect CUDA calls to CPU.
- The model is loaded **once** per run, then all pages are processed in a loop (loading is
  the slowest step, so never invoke the script once per page).

## Prerequisites (verify before running)

- Project home (has the model + the uv environment with torch / transformers / pymupdf /
  pillow-heif): `/Users/szou/Python/Playground/OCR`
  Override with env `UNLIMITED_OCR_HOME` if it moves.
- Model dir: `<HOME>/models/unlimited_ocr_official`. If missing, download it first:
  `huggingface-cli download baidu/Unlimited-OCR --local-dir <HOME>/models/unlimited_ocr_official`
- This is a **project-scoped** skill at `<HOME>/.claude/skills/unlimited-ocr/`. Run it from
  the project root so it uses that project's uv env (that's where the deps live).

## How to run

Single command for every input type. `<input>` is a single image, a directory of images,
or a PDF:

```bash
cd /Users/szou/Python/Playground/OCR     # project: holds the model, uv env, and this skill
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py <input> [options]
```

Examples:

```bash
# 1) single image (png/jpg/webp/tiff/bmp, or iPhone .heic)
... ocr.py /path/to/receipt.heic --no-crop

# 2) a directory of page images (sorted naturally: page1, page2, ...)
... ocr.py /path/to/scans/

# 3) a PDF — e.g. a scanned book (each page rendered, then OCR'd)
... ocr.py /path/to/book.pdf

# only some PDF pages, faster render
... ocr.py /path/to/book.pdf --pages 1-20 --dpi 150
```

## Options that matter

- `--no-crop` — **~40× faster on CPU** (≈6 s vs ≈250 s per page in tests). Does a single
  resized pass instead of high-resolution tiling. Use it for ordinary photos, receipts,
  screenshots, and slides. Keep the default (crop on) only for **dense, small-font, or
  high-resolution** pages (e.g. a textbook scan) where you need every character.
- `--split` — also emit one standalone Markdown per page at the top level
  (`outputs/<name>/page_0001.md`, `page_0002.md`, …) in addition to the combined `result.md`.
  Use when the downstream step processes pages one at a time (e.g. feeding each page to an LLM).
- `--out NAME` — output subdirectory name (defaults to the input's filename).
- `--pages 1-5,8` — for PDFs, OCR only these pages (1-based). Great for sampling a big book.
- `--dpi 180` — PDF render resolution. Lower (120–150) is faster; higher is sharper.
- `--prompt "document parsing."` — the task prompt (the script prepends `<image>`).
  Alternatives the model understands: `"Free OCR."` (plain text, no layout), `"Parse the figure."`.
- `--model-dir` / env `UNLIMITED_OCR_MODEL_DIR`, `--out-root` / env `UNLIMITED_OCR_OUT`.

## Output layout

Every run produces BOTH a combined file and per-page files, so the caller can pick either:

```
outputs/<name>/
  result.md                       # combined: all pages merged with page headers (whole-book context)
  page_0001.md, page_0002.md ...  # one standalone md per page — ONLY when --split is passed
  pages/page_0001/
    result.md                     # this page's Markdown (always written, nested with its assets)
    result_with_boxes.jpg         # the page annotated with detected layout boxes
    images/0.jpg ...              # figures/regions the model cropped out (referenced by the md)
  pages/page_0002/ ...
```

So "one md per page" is always available at `pages/page_NNNN/result.md`; `--split` additionally
gives clean, flat top-level `page_NNNN.md` files that are easier to iterate over.

After running, tell the user where `result.md` is and surface the recognized text. For a
multi-page job, `result.md` is the combined document; per-page detail lives under `pages/`.

## Expectations & tips

- **Speed is CPU-bound.** A scanned book of hundreds of pages with crop on can take hours.
  For long jobs, start with `--pages 1-3 --no-crop` to sanity-check quality, then scale up.
  `result.md` is written incrementally, so an interrupted run keeps completed pages.
- **HEIC/HEIF** (iPhone photos) are converted to PNG automatically (EXIF orientation applied).
- The model emits `<|det|>…<|/det|>` layout tags in the raw stream; the saved `result.md`
  is the cleaned version with figures linked as `![](…)`.
- If a region is non-text (a photo/figure), the model crops it into `images/` and links it
  rather than transcribing — that is expected layout behavior, not an error.

## Troubleshooting

- Empty output / first token is EOS → you're accidentally on MPS/CUDA. This script forces
  CPU; don't add `.to("mps")`. Don't "fix" it by editing files under `models/`.
- `模型目录不存在` → download the model (see Prerequisites) or pass `--model-dir`.
- HEIC error about `pillow-heif` → `uv add pillow-heif` in the project, then retry.
