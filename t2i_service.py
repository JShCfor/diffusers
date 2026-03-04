#!/usr/bin/env python3
"""文生图 Web 服务 — 基于 Gradio + diffusers"""

import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ─── 配置 ────────────────────────────────────────────────────────────────────
MODEL_ID   = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", 7860))
SHARE      = os.environ.get("SHARE", "false").lower() == "true"   # 公网分享

# ─── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"[加载] 模型: {MODEL_ID}")
print("[加载] 首次运行将自动下载模型，请耐心等待...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

print(f"[就绪] 设备: {device.upper()}  |  模型加载完成\n")

# ─── 推理函数 ─────────────────────────────────────────────────────────────────
def generate(prompt, negative_prompt, steps, guidance, width, height, seed, batch):
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed >= 0 else None
    result = pipe(
        prompt          = prompt,
        negative_prompt = negative_prompt or None,
        num_inference_steps = int(steps),
        guidance_scale  = guidance,
        width           = int(width),
        height          = int(height),
        num_images_per_prompt = int(batch),
        generator       = generator,
    )
    return result.images

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="文生图服务", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 文生图服务\nStable Diffusion v1.5 · 本地部署")

    with gr.Row():
        with gr.Column(scale=1):
            prompt   = gr.Textbox(label="提示词 (Prompt)", placeholder="a cat in space, photorealistic", lines=3)
            negative = gr.Textbox(label="负面提示词 (Negative)", placeholder="blurry, low quality, ugly", lines=2,
                                  value="blurry, bad quality, deformed, watermark")
            with gr.Row():
                steps    = gr.Slider(1, 50, value=20, step=1,  label="推理步数")
                guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="引导强度")
            with gr.Row():
                width  = gr.Dropdown([256, 384, 512, 640, 768], value=512, label="宽度")
                height = gr.Dropdown([256, 384, 512, 640, 768], value=512, label="高度")
            with gr.Row():
                seed  = gr.Number(label="随机种子 (-1 随机)", value=-1, precision=0)
                batch = gr.Slider(1, 4, value=1, step=1, label="生成数量")
            btn = gr.Button("生成图片", variant="primary")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="生成结果", columns=2, height=520)

    btn.click(fn=generate,
              inputs=[prompt, negative, steps, guidance, width, height, seed, batch],
              outputs=gallery)

    gr.Examples(
        examples=[
            ["a majestic lion in the savanna, golden hour, cinematic lighting", "", 20, 7.5, 512, 512, 42, 1],
            ["夕阳下的富士山，水彩画风格，高细节", "blurry, bad quality", 25, 7.5, 512, 512, -1, 1],
            ["cyberpunk city at night, neon lights, rain, ultra detailed", "", 30, 8.0, 512, 512, -1, 2],
        ],
        inputs=[prompt, negative, steps, guidance, width, height, seed, batch],
    )

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, share=SHARE)
