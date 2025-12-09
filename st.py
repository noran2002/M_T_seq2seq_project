import streamlit as st
import numpy as np
import pickle
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.ops.metrics_impl import false_negatives

# ================================
# Google Drive File IDs
# ================================

FILES = {
    "model": "1xn24qXxu3tJt7b4RYMYuSccTrk0INKda",
    "eng_tokenizer": "1R5sNKLZML5AW6hKfzomUJm7xefp-SKKB",
    "arb_tokenizer": "1ZYcWX-OHCbkJxTsZ7-0ZRNZD0Vg38B3R",
    "meta": "11IbXPHeamAa9eiYXwHBdWLIXcvG8u9WT"
}

# Local file names
MODEL_PATH = "eng2arb_model.h5"
ENG_TOKENIZER_PATH = "eng_tokenizer.pkl"
ARB_TOKENIZER_PATH = "arb_tokenizer.pkl"
META_PATH = "meta.pkl"

# ================================
# Download files if not exist
# ================================

def download_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

download_if_missing(FILES["model"], MODEL_PATH)
download_if_missing(FILES["eng_tokenizer"], ENG_TOKENIZER_PATH)
download_if_missing(FILES["arb_tokenizer"], ARB_TOKENIZER_PATH)
download_if_missing(FILES["meta"], META_PATH)

# ================================
# Load model + tokenizers + meta
# ================================

model = load_model(MODEL_PATH)
eng_tokenizer = pickle.load(open(ENG_TOKENIZER_PATH, "rb"))
arb_tokenizer = pickle.load(open(ARB_TOKENIZER_PATH, "rb"))
meta = pickle.load(open(META_PATH, "rb"))

input_length  = int(meta["input_length"])
output_length = int(meta["output_length"])
latent_dim    = int(meta["latent_dim"])
input_vocab   = int(meta["input_vocab"])
output_vocab  = int(meta["output_vocab"])

# ================================
# Translation Function
# ================================

def translate_sentence(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=input_length, padding='post')

    pred = model.predict(seq)
    pred = np.argmax(pred[0], axis=1)

    result = []
    for idx in pred:
        word = arb_tokenizer.index_word.get(idx, "")
        if word == "<end>":
            break
        if word not in ("<start>", ""):
            result.append(word)

    return " ".join(result).strip()

# ================================
# Streamlit Interface
# ================================

st.title("English → Arabic Translator 🇬🇧➡️🇪🇬")

user_text = st.text_input("Enter English text:")

if st.button("Translate"):
    if user_text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        translation = translate_sentence(user_text)
        st.success("Translation:")
        st.write(translation)


#####streamlit run st_app.py ## at terminal
