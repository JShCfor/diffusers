#!/usr/bin/env python3
"""文生图 Web 服务 — 基于 Gradio + diffusers"""

import os
from datetime import datetime
from pathlib import Path
import torch
import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

# ─── 模型预设 ─────────────────────────────────────────────────────────────────
# MODEL 环境变量可选值: sd15 | sdxl | 或任意本地/HF路径
PRESETS = {
    "sd15": {
        "path":       os.environ.get("SD15_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        "pipeline":   StableDiffusionPipeline,
        "label":      "SD 1.5 (快速 · 512px)",
        "def_width":  512,
        "def_height": 512,
        "sizes":      [256, 384, 512, 640, 768],
    },
    "sdxl": {
        "path":       os.environ.get("SDXL_PATH", "./models/sdxl-base-1.0"),
        "pipeline":   StableDiffusionXLPipeline,
        "label":      "SDXL 1.0 (高质量 · 1024px)",
        "def_width":  1024,
        "def_height": 1024,
        "sizes":      [512, 768, 832, 1024],
        "variant":    "fp16",   # 仅加载 fp16 权重文件
    },
}

# ─── 启动配置 ─────────────────────────────────────────────────────────────────
HOST        = os.environ.get("HOST", "0.0.0.0")
PORT        = int(os.environ.get("PORT", 7860))
SHARE       = os.environ.get("SHARE", "false").lower() == "true"
DEFAULT_MDL = os.environ.get("MODEL", "sd15")   # 默认启动模型: sd15 | sdxl

# ─── 设备 ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"[设备] {device.upper()}" + (f" · {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

# ─── 模型加载（懒加载缓存）────────────────────────────────────────────────────
_pipe_cache: dict = {}

def load_pipe(model_key: str):
    if model_key in _pipe_cache:
        return _pipe_cache[model_key]

    preset = PRESETS[model_key]
    print(f"[加载] {preset['label']}  ←  {preset['path']}")
    load_kwargs = dict(torch_dtype=dtype)
    if "variant" in preset:
        load_kwargs["variant"] = preset["variant"]
    if preset["pipeline"] is StableDiffusionPipeline:
        load_kwargs["safety_checker"] = None
    pipe = preset["pipeline"].from_pretrained(preset["path"], **load_kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    _pipe_cache[model_key] = pipe
    print(f"[就绪] {preset['label']}\n")
    return pipe

# 预加载默认模型
load_pipe(DEFAULT_MDL)

# ─── 推理函数 ─────────────────────────────────────────────────────────────────
def generate(model_key, prompt, negative_prompt, steps, guidance, width, height, seed, batch):
    pipe = load_pipe(model_key)
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed >= 0 else None
    result = pipe(
        prompt                = prompt,
        negative_prompt       = negative_prompt or None,
        num_inference_steps   = int(steps),
        guidance_scale        = guidance,
        width                 = int(width),
        height                = int(height),
        num_images_per_prompt = int(batch),
        generator             = generator,
    )
    # 自动保存到 outputs/ 目录
    out_dir = Path("outputs") / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(result.images):
        fname = f"{ts}_{i:02d}.png"
        img.save(out_dir / fname)
        print(f"[保存] {out_dir / fname}")
    return result.images

# ─── 动态更新分辨率选项 ───────────────────────────────────────────────────────
def on_model_change(model_key):
    p = PRESETS[model_key]
    return (
        gr.Dropdown(choices=p["sizes"], value=p["def_width"]),
        gr.Dropdown(choices=p["sizes"], value=p["def_height"]),
    )

# ─── Gradio UI ────────────────────────────────────────────────────────────────
model_choices = [(v["label"], k) for k, v in PRESETS.items()]

with gr.Blocks(title="文生图服务") as demo:
    gr.Markdown("# 文生图服务\nStable Diffusion 本地部署")

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=model_choices,
                value=DEFAULT_MDL,
                label="模型选择",
            )
            prompt   = gr.Textbox(label="提示词 (Prompt)", placeholder="a cat in space, photorealistic", lines=3)
            negative = gr.Textbox(label="负面提示词 (Negative)", lines=2,
                                  value="blurry, bad quality, deformed, watermark")
            with gr.Row():
                steps    = gr.Slider(1, 50, value=20, step=1,   label="推理步数")
                guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="引导强度")
            with gr.Row():
                p0 = PRESETS[DEFAULT_MDL]
                width  = gr.Dropdown(choices=p0["sizes"], value=p0["def_width"],  label="宽度",  allow_custom_value=True)
                height = gr.Dropdown(choices=p0["sizes"], value=p0["def_height"], label="高度",  allow_custom_value=True)
            with gr.Row():
                seed  = gr.Number(label="随机种子 (-1 随机)", value=-1, precision=0)
                batch = gr.Slider(1, 4, value=1, step=1, label="生成数量")
            btn = gr.Button("生成图片", variant="primary")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="生成结果", columns=2, height=560)

    # 切换模型时同步分辨率选项
    model_dd.change(fn=on_model_change, inputs=model_dd, outputs=[width, height])

    btn.click(
        fn=generate,
        inputs=[model_dd, prompt, negative, steps, guidance, width, height, seed, batch],
        outputs=gallery,
    )

    gr.Examples(
        examples=[
            ["sd15",  "a majestic lion in the savanna, golden hour, cinematic lighting", "", 20, 7.5, 512,  512,  42, 1],
            ["sdxl",  "夕阳下的富士山，水彩画风格，超高细节", "blurry, bad quality",     25, 7.5, 1024, 1024, -1, 1],
            ["sd15",  "cyberpunk city at night, neon lights, rain, ultra detailed",      "", 30, 8.0, 512,  512,  -1, 2],
        ],
        inputs=[model_dd, prompt, negative, steps, guidance, width, height, seed, batch],
    )

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, share=SHARE, theme=gr.themes.Soft())
