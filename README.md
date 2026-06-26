# Unlimited-OCR on Apple Silicon

Run Baidu's [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) document-parsing
model locally on a Mac (Apple Silicon), packaged as a reusable Claude Code skill that OCRs
**single images, folders of images, and multi-page scanned PDFs** into Markdown.

## Why CPU (not MPS)

Unlimited-OCR is a DeepSeek-V2 **MoE** model. Its expert routing is numerically wrong on the
PyTorch **MPS (Metal)** backend — inference returns empty output — and `PYTORCH_ENABLE_MPS_FALLBACK`
doesn't help (the ops are implemented-but-wrong, so they don't fall back). It runs **correctly on
CPU**, which is what this project uses. The official model code hardcodes `.cuda()`; we never edit
it — a small runtime monkeypatch in our scripts redirects CUDA calls to CPU instead.

## Setup

```bash
uv sync                                   # create the env (torch, transformers, pymupdf, pillow-heif…)

# download the model weights (~6.3 GB, gitignored) into models/
uv run huggingface-cli download baidu/Unlimited-OCR \
  --local-dir models/unlimited_ocr_official
```

## Usage

The skill lives at [`.claude/skills/unlimited-ocr/`](.claude/skills/unlimited-ocr/). Its bundled
script handles every input type and loads the model once per run:

```bash
# single image (png/jpg/webp/tiff/bmp, or iPhone .heic)
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py photo.heic --no-crop

# a directory of page images
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py ./scans/

# a multi-page scanned PDF (e.g. a book)
uv run python .claude/skills/unlimited-ocr/scripts/ocr.py book.pdf
```

The script is also self-contained (PEP 723), so `uv run` resolves its dependencies on its own.

Key flags: `--no-crop` (≈40× faster, single resized pass — good for ordinary photos; the default
high-res tiling is for dense/small-font pages), `--split` (also emit one `page_NNNN.md` per page),
`--pages 1-20` and `--dpi` (PDF sampling/resolution). See the skill's `SKILL.md` for the full guide.

## Output

```
outputs/<name>/
  result.md            # combined: all pages merged
  page_0001.md ...     # one md per page (with --split)
  pages/page_0001/     # per-page detail: result.md, result_with_boxes.jpg, images/
```

## Notes

- `models/`, `inputs/`, and `outputs/` are gitignored (large weights / personal scans / results).
- `crop_mode`, `base_size`, `image_size` are **native** Unlimited-OCR `infer()` parameters
  (the DeepSeek-OCR dynamic high-res tiling); this project only exposes them as CLI flags.
