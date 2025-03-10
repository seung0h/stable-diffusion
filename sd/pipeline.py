import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# uncond_prompt -> negative prompt that we don't want to generate
def generate(prompt: str, uncond_prompt: str, input_image=None, 
             strength=0.8, do_cfg=True, cfg_scale=7.5, 
             sampler="ddpm", n_inference_steps=50, models={}, 
             seed=None, device=None, idle_device=None, tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1]")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        # for deterministic generation by strictly setting seed in generator
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)
        
        # convert prompt into tokens using tokenizer
        if do_cfg:
            # consider uncond.
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # via CLIP -> (B, SEQ.(=77) 768)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # via CLIP -> (B, SEQ. 768)
            uncond_context = clip(uncond_tokens)

            # concat two context vectors: (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)

        to_idle(clip) # off-loaded clip for removing GPU burden

        # scheduler part
        if sampler == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (H, W, C) -> (B, H, W, C)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (B, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # more noise -> more creative (lossing input) controlled by strength
            latents = encoder(input_image_tensor, encoder_noise)

            # strength == 1 -> maximum noise (just gaussian)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # random start N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                # (B, 4, LATENTS_HEIGHT, LATENTS_WIDTH) -> (2*B, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise by Unet
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # remove noise predicted by Unet
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (B, C, H, W) -> (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x = x * (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

