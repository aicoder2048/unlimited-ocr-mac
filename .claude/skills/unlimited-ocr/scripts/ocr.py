#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch==2.10.0",
#     "torchvision==0.25.0",
#     "transformers==4.57.1",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "einops>=0.8.0",
#     "addict>=2.4.0",
#     "easydict>=1.13",
#     "pillow>=10.0.0",
#     "pillow-heif>=1.0.0",
#     "pymupdf>=1.26.0",
#     "matplotlib>=3.8.0",
#     "numpy>=1.24.0",
#     "requests>=2.30.0",
#     "psutil>=5.9.0",
# ]
# ///
"""
unlimited-ocr —— 用本地 baidu/Unlimited-OCR 模型批量做文档解析(Apple Silicon, CPU)。

自包含运行(PEP 723:uv 按上方内联依赖自动建环境,不依赖项目 pyproject):
    uv run scripts/ocr.py <输入> [选项]

支持三种输入:
  1) 单张图片         ocr.py page.png
  2) 图片目录         ocr.py ./scans/            (按文件名自然排序,逐张 OCR)
  3) PDF(扫描书等)  ocr.py book.pdf             (pymupdf 逐页渲染再 OCR)

输出(默认到 <项目>/outputs/<输入名>/):
  outputs/<name>/
    result.md                      <- 主交付物:全部页面合并的 Markdown(插图路径已修正)
    pages/page_001/
      result.md                    <- 单页 Markdown
      result_with_boxes.jpg        <- 带版面框的标注图
      images/0.jpg ...             <- 模型抽出的插图(被 md 引用)

为什么 CPU:官方是 DeepSeek-V2 MoE,其专家路由在 PyTorch MPS 后端上数值算错(实测输出为空),
CPU 是已验证正确的路径。本脚本用 monkeypatch 把官方 remote code 写死的 .cuda() 重定向到 CPU,
不修改 models/ 下任何官方文件。模型只加载一次,之后循环处理所有页,避免重复加载(加载是最慢的一步)。
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- 项目/模型默认位置(可用环境变量或命令行覆盖)----
PROJECT_HOME = os.environ.get("UNLIMITED_OCR_HOME", "/Users/szou/Python/Playground/OCR")
DEFAULT_MODEL_DIR = os.environ.get(
    "UNLIMITED_OCR_MODEL_DIR", os.path.join(PROJECT_HOME, "models", "unlimited_ocr_official")
)
DEFAULT_OUT_ROOT = os.environ.get("UNLIMITED_OCR_OUT", os.path.join(PROJECT_HOME, "outputs"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
HEIC_EXTS = {".heic", ".heif"}


# =================== monkeypatch:CUDA -> CPU(不改官方文件)===================
def _patch_torch_to_cpu():
    import torch

    torch.Tensor.cuda = lambda self, *a, **k: self          # x.cuda()      -> 留在 CPU
    torch.nn.Module.cuda = lambda self, *a, **k: self        # module.cuda() -> no-op
    torch.cuda.is_available = lambda: False
    _orig_autocast = torch.autocast

    def _cpu_safe_autocast(device_type="cuda", *args, **kwargs):
        if device_type == "cuda":                            # autocast("cuda") 在 CPU 上禁用
            kwargs["enabled"] = False
        return _orig_autocast(device_type, *args, **kwargs)

    torch.autocast = _cpu_safe_autocast
    return torch


def normalize_image(path, work_dir):
    """HEIC/HEIF(如 iPhone 照片)-> PNG 并应用 EXIF 方向;其它格式原样返回。
    模型内部用 PIL.Image.open 读图,PIL 原生不支持 HEIC,故先转码再喂给模型。"""
    if os.path.splitext(path)[1].lower() not in HEIC_EXTS:
        return path
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        sys.exit("[error] 检测到 HEIC/HEIF,但缺少 pillow-heif。请先安装: uv add pillow-heif")
    from PIL import Image, ImageOps

    os.makedirs(work_dir, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    out = os.path.join(work_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
    img.save(out)
    return out


def load_engine(model_dir):
    """加载一次模型 + tokenizer(返回 model, tokenizer)。"""
    if not os.path.isdir(model_dir):
        sys.exit(
            f"[error] 模型目录不存在: {model_dir}\n"
            f"        请先下载: huggingface-cli download baidu/Unlimited-OCR --local-dir <dir>\n"
            f"        或用 --model-dir / 环境变量 UNLIMITED_OCR_MODEL_DIR 指定。"
        )
    torch = _patch_torch_to_cpu()
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    print(f"[info] 加载模型 (CPU, bf16): {model_dir}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"   # use_mla=False -> 仅 mha_eager 有效;Mac 无 flash-attn
    model = AutoModel.from_pretrained(
        model_dir, config=config, trust_remote_code=True,
        use_safetensors=True, dtype=torch.bfloat16,
    ).eval()
    print(f"[info] 模型就绪,用时 {time.time() - t0:.0f}s", flush=True)
    return model, tokenizer


# =================== 输入解析 ===================
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def collect_pages(input_path, dpi, tmp_dir, pages_spec=None):
    """把输入展开成有序的图片路径列表 [(page_no, image_path), ...]。"""
    p = Path(input_path)
    if not p.exists():
        sys.exit(f"[error] 输入不存在: {input_path}")

    if p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS], key=_natural_key)
        if not imgs:
            sys.exit(f"[error] 目录里没有图片: {input_path}")
        return [(i + 1, str(f)) for i, f in enumerate(imgs)]

    if p.suffix.lower() == ".pdf":
        import fitz  # pymupdf

        doc = fitz.open(str(p))
        wanted = _parse_pages_spec(pages_spec, len(doc))
        out = []
        os.makedirs(tmp_dir, exist_ok=True)
        for idx in wanted:
            page = doc[idx]
            pix = page.get_pixmap(dpi=dpi)
            img_path = os.path.join(tmp_dir, f"page_{idx + 1:04d}.png")
            pix.save(img_path)
            out.append((idx + 1, img_path))
        doc.close()
        if not out:
            sys.exit("[error] PDF 没有可处理的页面")
        return out

    if p.suffix.lower() in IMAGE_EXTS:
        return [(1, str(p))]

    sys.exit(f"[error] 不支持的输入类型: {input_path}(支持单图 / 图片目录 / .pdf)")


def _parse_pages_spec(spec, n):
    """'1-5,8,10-' -> 0 基索引列表;None -> 全部。"""
    if not spec:
        return list(range(n))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1
            b = int(b) if b else n
            out.update(range(a - 1, b))
        elif part:
            out.add(int(part) - 1)
    return sorted(i for i in out if 0 <= i < n)


# =================== OCR ===================
def ocr_page(model, tokenizer, image_path, page_dir, prompt, base_size, image_size, crop_mode, max_length):
    """对单张图调用官方 infer(save_results 模式),返回该页 result.md 文本。"""
    os.makedirs(page_dir, exist_ok=True)
    if "<image>" not in prompt:
        prompt = "<image>" + prompt
    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=page_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=max_length,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )
    md_path = os.path.join(page_dir, "result.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR 批量文档解析 (CPU, Apple Silicon)")
    ap.add_argument("input", help="单张图片 / 图片目录 / PDF 路径")
    ap.add_argument("--out", default=None, help="输出子目录名(默认取输入文件名)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help=f"输出根目录(默认 {DEFAULT_OUT_ROOT})")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    ap.add_argument("--prompt", default="document parsing.", help="OCR 提示词")
    ap.add_argument("--pages", default=None, help="仅处理 PDF 指定页,如 '1-5,8'(1 基)")
    ap.add_argument("--dpi", type=int, default=180, help="PDF 渲染 DPI(默认 180)")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--no-crop", action="store_true", help="关闭动态切片(整页一次)")
    ap.add_argument("--split", action="store_true",
                    help="每页额外输出一个独立 md(顶层 page_0001.md, page_0002.md…),便于逐页喂 AI")
    ap.add_argument("--max-length", type=int, default=8192)
    args = ap.parse_args()

    in_path = Path(args.input)
    name = args.out or re.sub(r"[^\w.\-]+", "_", in_path.stem if in_path.suffix else in_path.name)
    out_dir = os.path.join(args.out_root, name)
    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    pages = collect_pages(args.input, args.dpi, tmp_dir=os.path.join(out_dir, "_pdf_pages"),
                          pages_spec=args.pages)
    print(f"[info] 输入: {args.input}  ->  {len(pages)} 页  ->  {out_dir}", flush=True)

    model, tokenizer = load_engine(args.model_dir)

    combined_path = os.path.join(out_dir, "result.md")
    combined = open(combined_path, "w", encoding="utf-8")
    combined.write(f"# OCR: {name}\n\n")

    t_all = time.time()
    convert_dir = os.path.join(out_dir, "_converted")
    for n, (page_no, img) in enumerate(pages, 1):
        page_dir = os.path.join(pages_dir, f"page_{page_no:04d}")
        img = normalize_image(img, convert_dir)   # HEIC/HEIF -> PNG(其它格式无操作)
        t0 = time.time()
        print(f"[page {n}/{len(pages)}] OCR {img} ...", flush=True)
        try:
            text = ocr_page(model, tokenizer, img, page_dir, args.prompt,
                            args.base_size, args.image_size, not args.no_crop, args.max_length)
        except Exception as e:  # 单页失败不影响整本
            print(f"  [warn] 第 {page_no} 页失败: {e!r}", flush=True)
            text = f"_[OCR failed: {e!r}]_"
        # 修正插图相对路径:images/N.jpg -> pages/page_NNNN/images/N.jpg
        text = text.replace("](images/", f"](pages/page_{page_no:04d}/images/")
        if args.split:  # 每页一个独立 md(顶层平铺,插图路径已指向 pages/…,可直接渲染)
            with open(os.path.join(out_dir, f"page_{page_no:04d}.md"), "w", encoding="utf-8") as pf:
                pf.write(text + "\n")
        if len(pages) > 1:
            combined.write(f"\n\n---\n\n## Page {page_no}\n\n")
        combined.write(text + "\n")
        combined.flush()  # 增量落盘,长文档中断也保留已完成部分
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    combined.close()
    print(f"\n[done] {len(pages)} 页, 共 {time.time() - t_all:.0f}s", flush=True)
    print(f"[done] 合并结果: {combined_path}", flush=True)
    if args.split:
        print(f"[done] 每页独立 md: {out_dir}/page_*.md", flush=True)
    print(f"[done] 每页明细(带框图/插图): {pages_dir}/page_*/", flush=True)


if __name__ == "__main__":
    main()
