import streamlit as st
from transformers import pipeline
st.write("Testing transformers")
model = pipeline("text-generation")
result = model("Hello, world!")[0]["generated_text"]
st.write(result)