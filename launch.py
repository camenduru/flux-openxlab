import random
import torch
import numpy as np
from PIL import Image
import nodes
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management
import gradio as gr

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

with torch.inference_mode():
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-dev.sft", "default")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(positive_prompt, width, height, seed, steps, sampler_name, scheduler):
    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(seed)
    cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    noise = RandomNoise.get_noise(seed)[0] 
    guider = BasicGuider.get_guider(unet, cond)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
    latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
    sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    model_management.soft_empty_cache()
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/home/xlab-app-center/flux.png")
    return "/home/xlab-app-center/flux.png"

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            positive_prompt = gr.Textbox(lines=3, interactive=True, value="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black dress with a gold leaf pattern and a white apron eating a slice of an apple pie in the kitchen of an old dark victorian mansion with a bright window and very expensive stuff everywhere", label="Prompt")
            width = gr.Slider(minimum=256, maximum=2048, value=1024, step=16, label="width")
            height = gr.Slider(minimum=256, maximum=2048, value=1024, step=16, label="height")
            seed = gr.Slider(minimum=0, maximum=18446744073709551615, value=0, step=1, label="seed (0=random)")
            steps = gr.Slider(minimum=4, maximum=50, value=20, step=1, label="steps")
            sampler_name = gr.Dropdown(["euler", "heun", "heunpp2", "heunpp2", "dpm_2", "lms", "dpmpp_2m", "ipndm", "deis", "ddim", "uni_pc", "uni_pc_bh2"], label="sampler_name", value="euler")
            scheduler = gr.Dropdown(["normal", "sgm_uniform", "simple", "ddim_uniform"], label="scheduler", value="simple")
            generate_button = gr.Button("Generate")
        with gr.Column():
            output_image = gr.Image(label="Generated image", interactive=False)

    generate_button.click(fn=generate, inputs=[positive_prompt, width, height, seed, steps, sampler_name, scheduler], outputs=output_image)

demo.launch(inline=False, share=False, debug=True)