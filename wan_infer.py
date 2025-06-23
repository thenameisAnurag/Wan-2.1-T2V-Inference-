
import torch
import os 
import gc

# Set visible GPUs to 1,2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# Load VAE on CPU first
vae = AutoencoderKLWan.from_pretrained(
    model_id, 
    subfolder="vae", 
    torch_dtype=torch.float32
)

flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction', 
    use_flow_sigmas=True, 
    num_train_timesteps=1000, 
    flow_shift=flow_shift
)

# Load pipeline on CPU first
pipe = WanPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

pipe.scheduler = scheduler

# Enable CPU offloading to manage memory across GPUs
pipe.enable_model_cpu_offload(gpu_id=0)  # This will use the first GPU (CUDA:0)

# Enable attention slicing only (skip xformers)
pipe.enable_attention_slicing(slice_size=1)

# Enable VAE slicing if available
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()

# Use a slightly shorter prompt and smaller dimensions to help with memory
prompt = "A stylish man sips espresso outside a Paris café, dressed in a tailored camel trench coat, black turtleneck, slim-fit trousers, and leather loafers. A beret rests on his head as he reads a fashion magazine. The cobblestone street, wrought-iron chairs, and blooming flowers enhance the chic yet casual Parisian vibe."
negative_prompt = "Avoid sloppy, unkempt appearance; no casual t-shirts, baggy jeans, or sneakers. Not wearing athletic wear, baseball caps, or bright neon colors. No tropical print shirts or cargo shorts. Setting is not a fast food restaurant, shopping mall, or suburban street. Not holding a smartphone or taking selfies. No crowded tourist attractions or modern glass buildings in background. Avoid plastic furniture, concrete sidewalks, or chain store signage. Not wearing sunglasses or headphones. No busy traffic or parked cars visible. Not a rainy or overcast day. No neon lighting or nightclub atmosphere. Not surrounded by tourists with cameras. No modern technology visible. Not casual American streetwear aesthetic."

# Reduce dimensions to avoid OOM
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=720,  
    width=1280,   
    num_frames=81,  
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=16)