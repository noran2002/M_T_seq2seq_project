
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import path



BASE_DIR = "D:\My Drive"
MODEL_PATH = path.join(BASE_DIR, "eng2arb_model.h5")
ENG_TOKENIZER_PATH = path.join(BASE_DIR, "eng_tokenizer.pkl")
ARB_TOKENIZER_PATH = path.join(BASE_DIR, "arb_tokenizer.pkl")
META_PATH = path.join(BASE_DIR, "meta.pkl")

# -----------------------------

# Load model + tokenizers + meta

# -----------------------------

model = load_model(MODEL_PATH)
eng_tokenizer = pickle.load(open(ENG_TOKENIZER_PATH, "rb"))
arb_tokenizer = pickle.load(open(ARB_TOKENIZER_PATH, "rb"))
meta = pickle.load(open(META_PATH, "rb"))

input_length  = int(meta["input_length"])
output_length = int(meta["output_length"])
latent_dim    = int(meta["latent_dim"])
input_vocab   = int(meta["input_vocab"])
output_vocab  = int(meta["output_vocab"])

# -----------------------------

# Function: translate English → Arabic

# -----------------------------

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


# -----------------------------

# Streamlit Page

# -----------------------------

st.title("English → Arabic Translator 🇬🇧➡️🇦🇪")

user_text = st.text_input("Enter English text:")

if st.button("Translate"):
    if user_text.strip() == "":
        st.warning("Please write a sentence to translate.")
    else:
        translation = translate_sentence(user_text)
        st.success("Translation:")
        st.write(translation)


#####streamlit run st_app.py ## at terminal