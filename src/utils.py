import os
import google.generativeai as genai
import logging
import requests
import json

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_prompt():
    input_prompt = """
                   You are tool for image captioning and generate prompt for Stable Diffusion AnyLoRA.
                   
                   Example prompt: (young Asian woman:1.5), (25 years old:1.3), (oval face:1.3), (clear skin:1.3), (warm complexion:1.2), (bright almond-shaped eyes:1.4), (natural makeup:1.2), (defined eyebrows:1.2), (friendly smile:1.3), (straight white teeth:1.2), (small nose:1.2), (black hair pulled back:1.4), (neat bun hairstyle:1.3), (slender neck:1.2), (light beige hooded t-shirt:1.3), (small yellow pin on shirt:1.2), (athletic build:1.2), (good posture:1.3), (standing in an office:1.3), (small tattoo on left arm:1.1), (professional appearance:1.3)
                   
                   Please generate prompt for Stable Diffusion including weight on each prominent point and response just only prompt text.
                   """
    return input_prompt

def prompt_with_llm(input_question, prompt, image):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([input_question, prompt, image])
    return f"""
            (best quality:1.5), (solo:1.4), (upper body:1.4), (looking at viewer:1.4),
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
        "negative_prompt": "lowres, blurry, worst quality, low quality, normal quality, many people, bad anatomy, bad hands, missing fingers, error, text, username, extra digit, fewer digits, signature, watermark, cropped, jpeg artifacts",
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
        "send_images": True
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = json.loads(response.text)
    logging.info(f"generate_image.code: {str(response.status_code)}", exc_info=True)
    # logging.info(f"generate_image.text: {str(response.text)}", exc_info=True)
    
    return result