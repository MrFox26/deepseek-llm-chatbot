
import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Streamlit App Title and Description
st.title("DeepSeek LLM Chatbot")
st.write("Generate responses using the DeepSeek LLM-7B model.")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

# Initialize Model and Tokenizer
tokenizer, model = load_model()

# File Listing (if required)
st.write("### Available Files in Input Directory")
input_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        input_files.append(file_path)
if input_files:
    st.write(input_files)
else:
    st.write("No files found in the input directory.")

# Text Input for User Prompt
user_input = st.text_area("Enter your prompt below:", height=150)

# Button to Generate Response
if st.button("Generate Text"):
    if user_input:
        # Tokenize and generate text
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display Generated Text
        st.subheader("Generated Response:")
        st.write(generated_text)
    else:
        st.warning("Please enter a valid prompt before generating text.")
