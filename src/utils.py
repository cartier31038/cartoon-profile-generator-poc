import base64
import google.generativeai as genai
import io
import json
import logging
import os
import requests

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_prompt():
    system_prompt = """
                   You are tool for image captioning and generate prompt for Stable Diffusion AnyLoRA.
                   
                   Example prompt: (young Asian woman:1.5), (25 years old:1.3), (oval face:1.3), (clear skin:1.3), (warm complexion:1.2), (bright almond-shaped eyes:1.4), (natural makeup:1.2), (defined eyebrows:1.2), (friendly smile:1.3), (straight white teeth:1.2), (small nose:1.2), (black hair pulled back:1.4), (neat bun hairstyle:1.3), (slender neck:1.2), (light beige hooded t-shirt:1.3), (small yellow pin on shirt:1.2), (good posture:1.3), (small tattoo on left arm:1.1), (professional appearance:1.3)
                   
                   Please suggest a prompt for Stable Diffusion from the attached image and describe only the person's appearance in as much detail as you can, without including the background.
                   """
                   
    return system_prompt

def prompt_with_llm(user_prompt, system_prompt, image):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([user_prompt, system_prompt, image])
    return f"""
            (best quality:1.5), (white background:1.5), (solo:1.4), (upper body:1.4), (looking at viewer:1.4),
            {response.text}
            <lora:last:0.9>
            """
            
def generate_image(prompt):
    
    url = f"{os.getenv('SD_API_URL')}/sdapi/v1/txt2img"
    headers = {
        "Content-Type": "application/json",
    }

    logging.info(f"generate_image.prompt: {prompt}", exc_info=True)
    
    data = {
        "prompt": prompt,
        "negative_prompt": "lowres, blurry, worst quality, low quality, normal quality, many people, bad anatomy, bad hands, missing fingers, error, text, username, extra digit, fewer digits, signature, watermark, cropped, jpeg artifacts, detailed background, glitch rim",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 30,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": False,
        "tiling": False,
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "sampler_index": "Euler a",
        "save_images": False,
        "send_images": True,
        "denoising_strength": 0.75
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = json.loads(response.text)
    logging.info(f"generate_image.code: {str(response.status_code)}", exc_info=True)
    # logging.info(f"generate_image.text: {str(response.text)}", exc_info=True)
    
    return result

def generate_img2img(prompt, image):
    
    url = f"{os.getenv('SD_API_URL')}/sdapi/v1/img2img"

    headers = {
        "Content-Type": "application/json",
    }

    logging.info(f"generate_img2img.prompt: {prompt}", exc_info=True)
    # logging.info(f"generate_img2img.img: {image}", exc_info=True)
    
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    bytes_image = byte_stream.getvalue()
    encoded_image = base64.b64encode(bytes_image).decode('utf-8')

    logging.info(f"generate_img2img.blob_image: {encoded_image}", exc_info=True)
    data = {
        "prompt": prompt,
        "negative_prompt": "lowres, blurry, worst quality, low quality, normal quality, many people, bad anatomy, bad hands, missing fingers, error, text, username, extra digit, fewer digits, signature, watermark, cropped, jpeg artifacts, detailed background, glitch rim",
        "batch_size": 3,
        "n_iter": 1,
        "steps": 60,
        "cfg_scale": 8,
        "width": 512,
        "height": 768,
        "restore_faces": False,
        "tiling": False,
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "sampler_index": "Euler a",
        "save_images": False,
        "send_images": True,
        "denoising_strength": 0.8,
        "include_init_images": True,
        "init_images": [
            encoded_image
        ],
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = json.loads(response.text)
    logging.info(f"generate_img2img.code: {str(response.status_code)}", exc_info=True)
    # logging.info(f"generate_img2img.text: {str(response.text)}", exc_info=True)
    
    return result