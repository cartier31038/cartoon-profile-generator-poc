import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_prompt():
    input_prompt = """
                   You are tool for image captioning and generate prompt for Stable Diffusion .
                   Example prompt: Headshot of a young Southeast Asian man (weight: 1.4), friendly smile (weight: 1.2), short brown hair styled upwards (weight: 1.3), wearing a dark gray t-shirt with 'Los White Sox' logo and emblem (weight: 1.5), small hoop earrings (weight: 1.1), against a plain off-white background (weight: 1.2). Clear skin (weight: 1.1), warm complexion (weight: 1.2), direct eye contact (weight: 1.3), shoulders visible (weight: 1.1), soft indoor lighting (weight: 1.0), casual and approachable demeanor (weight: 1.2), centered composition (weight: 1.1).
                   Please generate prompt for Stable Diffusion including weight on each prominent point and response just only prompt text.
                   """
    return input_prompt

def prompt_with_llm(input_question, prompt, image):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([input_question, prompt, image])
    return f"""
            (best quality:1.1), solo, upper body, looking at viewer, (solid white background:1.1),
            {response.text} 
            \<lora:last:0.9\>
            """