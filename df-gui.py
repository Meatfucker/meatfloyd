from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
import random
import tkinter as tk
from PIL import Image, ImageTk

# Set up the DeepFloyd IF pipeline
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()
stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
stage_2.enable_model_cpu_offload()
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
stage_3.enable_model_cpu_offload()


# Create the main window
window = tk.Tk()
window.title("DeepFloyd IF")

# Load and display the image
image = Image.open("default.png")
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(window, image=photo)
image_label.pack(side="left")

# Create frame for buttons and inputs
input_frame = tk.Frame(window)
input_frame.pack(side="right")

# Create the text input box
promptfield = tk.Entry(input_frame)
promptfield.pack()

# Create the generate button
def generate():
    # Insert your arbitrary Python code here
    # text embeds
    prompt = promptfield.get()
    print(f"Prompt contains: {prompt}")
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    
    # randomize seed
    randseed = random.randrange(6500000)
    generator = torch.manual_seed(randseed)

    # Call pipeline
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    
    # Save and display image
    image[0].save(f"./outputs/{prompt}_{randseed}.png")
    new_image = Image.open(f"./outputs/{prompt}_{randseed}.png")
    photo = ImageTk.PhotoImage(new_image)
    image_label.configure(image=photo)
    image_label.image = photo  # This line is needed to prevent the photo from being garbage collected
    print("Generate button clicked!")

generate_button = tk.Button(input_frame, text="Generate", command=generate)
generate_button.pack()

# Start the event loop
window.mainloop()
