
import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Streamlit App Title and Description
st.title("DeepSeek LLM Chatbot")
st.write("Generate responses using the DeepSeek LLM-7B model")

# Model Initialization
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Input Section for User Prompt
user_input = st.text_area("Enter your prompt below:", height=150)

# File Listing (if required)
st.write("Available files in the input directory:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        st.write(os.path.join(dirname, filename))

# Button to Generate Text
if st.button("Generate Text"):
    if user_input:
        # Tokenize and generate response
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display Generated Text
        st.subheader("Generated Response:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt before clicking generate.")
