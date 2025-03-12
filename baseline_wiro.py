from diffusers import FluxPipeline
import torch

pipeline = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('WiroAI/Pokemon-Universe-Flux-Dev-fp8', weight_name='pokemon-universe-flux-dev-fp8.safetensors')
image = pipeline('lgdsgn, A moth Pokémon with glass-like wings that reflect viewers’ memories; it shatters and reforms endlessly, revealing hidden truths.').images[0]
image.save("baseline_wiro.png")