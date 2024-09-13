import streamlit as st
from PIL import Image
from utils import load_prompt, prompt_with_llm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    st.set_page_config("Cartoon Profile Generator ( ^▽^)σ")

    st.title("Cartoon Profile Generator ( ^▽^)σ")

    user_question = st.text_input("Input prompt", key="input")

    st.sidebar.title("Profile Image")

    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=["jpg", "png", "jpeg"]
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)
            logging.info(f"Image uploaded successfully. Size: {image.size}, Format: {image.format}")
        except Exception as e:
            st.error(f"Error opening the image: {str(e)}")
            logging.error(f"Error opening the image: {str(e)}")

    prompt = load_prompt()

    if st.button("Generate"):
        if image is None:
            st.warning("Please upload an image before generating.")
            return

        with st.spinner("Cartoon profile generating..."):
            try:
                response = prompt_with_llm(input_question=user_question, prompt=prompt, image=image)
                st.subheader("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
                logging.error(f"Error in prompt_with_llm: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()