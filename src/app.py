import base64
import io
import logging
import streamlit as st

from PIL import Image
from utils import load_prompt, prompt_with_llm, generate_image, generate_img2img

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    st.set_page_config("Cartoon Profile Generator ( ^▽^)σ")

    user_prompt = st.text_input("Input prompt", placeholder="Enter your specific appearance.", key="input")
    
    st.sidebar.title("Cartoon Profile Generator\n( ^▽^)σ")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a your reference profile...", 
        type=["jpg", "png", "jpeg"]
    )

    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Uploaded image", use_column_width=True)
            logging.info(f"Image uploaded successfully. Size: {image.size}, Format: {image.format}")
        except Exception as e:
            st.error(f"Error opening the image: {str(e)}")
            logging.error(f"Error opening the image: {str(e)}")
    else:
        image = None

    if st.button("Generate profile"):
        if image is None:
            st.warning("Please upload an image before generating.")
            return

        with st.spinner("Profile generating..."):
            try:
                prompt = load_prompt()
                caption = prompt_with_llm(user_prompt=user_prompt, system_prompt=prompt, image=image)
                st.subheader("Caption:")
                st.write(caption)
                
                # result = generate_image(caption)
                result = generate_img2img(caption, image)
                length = len(result['images'])
                
                with col2:
                    cols = st.columns(length)
                    for i, image in enumerate(result['images']):
                        image_bytes = base64.b64decode(image)
                        final_image = Image.open(io.BytesIO(image_bytes))
                        with cols[i % length]:
                            st.image(final_image, caption="Result image", use_column_width=True)
        
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
                logging.error(f"An error occurred during generation: {str(e)}", exc_info=True)
    
if __name__ == "__main__":
    main()