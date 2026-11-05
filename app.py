import sys
# Polyfill for Python 3.13+ missing cgi module required by older httpx/googletrans versions
try:
    import cgi
except ImportError:
    import html
    import sys
    from types import ModuleType
    cgi_module = ModuleType("cgi")
    cgi_module.escape = html.escape
    sys.modules["cgi"] = cgi_module

import os
from gtts import gTTS 
from googletrans import Translator, LANGUAGES 
import streamlit as st 

# Initialize Translator
translator = Translator()

# Page Config
st.set_page_config(
    page_title="Language Translation Tool",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 AI Language Translation Tool")
st.write(
    "Translate text seamlessly between multiple languages."
)

# Create mapping for language selection
lang_codes = {name.capitalize(): code for code, name in LANGUAGES.items()}
sorted_languages = sorted(lang_codes.keys())

# Main Interface Elements
col1, col2 = st.columns(2)
with col1:
    src_lang_name = st.selectbox(
        "Source Language", ["Auto Detect"] + sorted_languages
    )
with col2:
    tgt_lang_name = st.selectbox(
        "Target Language", sorted_languages, index=sorted_languages.index("English")
    )

# Text Input Area
text_to_translate = st.text_area(
    "Enter text to translate:",
    placeholder="Type something here...",
    height=150,
)

if st.button("Translate Text", type="primary"):
    if text_to_translate.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        try:
            # Map selection back to codes
            src_code = (
                "auto" if src_lang_name == "Auto Detect" else lang_codes[src_lang_name]
            )
            tgt_code = lang_codes[tgt_lang_name]

            # Perform Translation
            translation = translator.translate(
                text_to_translate, src=src_code, dest=tgt_code
            )
            st.success("Translation Successful!")

            # Display Result
            st.markdown("### Translated Text:")
            st.info(translation.text)

            # Optional Feature: Text-to-Speech (Audio Output)
            try:
                tts = gTTS(text=translation.text, lang=tgt_code, slow=False)
                audio_file = "translated_audio.mp3"
                tts.save(audio_file)
                st.audio(audio_file, format="audio/mp3")
            except Exception as audio_error:
                st.info(
                    "Text-to-speech audio generation is currently unavailable for this"
                    " target language."
                )

        except Exception as e:
            st.error(
                f"An error occurred during translation: {e}. Please check your internet"
                " connection."
            )
