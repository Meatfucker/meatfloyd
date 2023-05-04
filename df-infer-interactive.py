from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
import random
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import messagebox

prompt = ""
# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
stage_2.enable_model_cpu_offload()

# stage 3
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
stage_3.enable_model_cpu_offload()

while True:
    
    # Set up interactive prompt
    oldprompt = prompt
    prompt = input("Enter prompt text, 'q' to quit, or 'r' to reroll: ")
    if prompt == 'q':
        break
    if prompt == 'r':
        prompt = oldprompt
    
    # text embeds
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    
    # randomize seed
    randseed = random.randrange(6500000)
    generator = torch.manual_seed(randseed)

    # stage 1
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    
    # stage 2
    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    
    # stage 3
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    
    # Save and display image
    image[0].save(f"./outputs/{prompt}_{randseed}.png")
    img = Image.open(f"./outputs/{prompt}_{randseed}.png")
    root = tk.Tk()
    root.title("DeepFloydIF")
    img_label = tk.Label(root)
    img_label = tk.Label(root)
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.pack()
    root.mainloop()
   