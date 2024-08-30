import streamlit as st
from helper import decode_sentence, load_transformer_model
import string

# Initialize model and text preprocessing objects
transformer = load_transformer_model("french_tamil_transformer.keras")

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

# Streamlit app
st.title("Language Translation")

input_text = st.text_area("Enter a sentence to translate:", "")
output_placeholder=st.empty()
if st.button("Translate"):
    input_sentence = input_text.lower()
    if len(input_sentence.split()) == 5:
        input_sentence = input_sentence.translate(str.maketrans('', '', strip_chars))
        translated = decode_sentence(input_sentence, transformer)
        translated=translated.replace("[start] ","")
        output_html = f"""
        <div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;'>
            <p style='font-weight: bold;'>Translated Sentence:</p>
            <p>{translated}</p>
        </div>
        """
        # Display the translated sentence inside the bordered section
        output_placeholder.markdown(output_html, unsafe_allow_html=True)
    else:
        # Display the error message directly in the placeholder
        output_placeholder.write("Please provide exactly 5 words.")



