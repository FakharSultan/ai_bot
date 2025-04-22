import streamlit as st
import sqlite3
import json
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from TTS.api import TTS
import torch

# Initialize SQLite database for character memory
def init_db():
    conn = sqlite3.connect("characters.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS characters (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT UNIQUE,
                 traits TEXT,
                 appearance TEXT,
                 created_at TIMESTAMP
                 )""")
    conn.commit()
    conn.close()

# Save character to database
def save_character(name, traits, appearance):
    conn = sqlite3.connect("characters.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO characters (name, traits, appearance, created_at) VALUES (?, ?, ?, ?)",
              (name, traits, appearance, datetime.now()))
    conn.commit()
    conn.close()

# Retrieve character from database
def get_character(name):
    conn = sqlite3.connect("characters.db")
    c = conn.cursor()
    c.execute("SELECT name, traits, appearance FROM characters WHERE name = ?", (name,))
    result = c.fetchone()
    conn.close()
    return result

# Web search function (BeautifulSoup fallback, no SerpAPI key required)
def web_search(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.find_all("div", class_="BNeawe")
        return snippets[0].text if snippets else "No results found."
    except Exception as e:
        return f"Error searching: {str(e)}"

# Image generation using lightweight Stable Diffusion
def generate_image(prompt, output_path="output.png"):
    try:
        model_id = "stabilityai/stable-diffusion-2-1-base"  # Smaller model
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        pipe = pipe.to("cpu")  # Explicitly use CPU
        image = pipe(prompt, height=256, width=256).images[0]  # Low resolution
        image.save(output_path)
        return output_path
    except Exception as e:
        return f"Error generating image: {str(e)}"

# Text-to-speech using Coqui TTS
def text_to_speech(text, output_path="output.wav"):
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        tts.tts_to_file(text=text, file_path=output_path)
        return output_path
    except Exception as e:
        return f"Error generating speech: {str(e)}"

# Main chatbot logic
def chatbot(query, character_name=None):
    # Initialize lightweight LLM (DistilGPT2)
    try:
        llm = pipeline("text-generation", model="distilgpt2", device=-1)  # CPU only
    except Exception:
        llm = lambda x: [{"generated_text": "LLM not available. Using simple response."}]

    # Check for character memory
    character_info = ""
    if character_name:
        character = get_character(character_name)
        if character:
            name, traits, appearance = character
            character_info = f"Character: {name}, Traits: {traits}, Appearance: {appearance}\n"
        else:
            return f"Character '{character_name}' not found."

    # Handle different query types
    response = ""
    if "generate image" in query.lower():
        prompt = query.replace("generate image", "").strip()
        if character_name:
            prompt = f"{character_info} {prompt}"
        response = generate_image(prompt)
    elif "generate video" in query.lower():
        response = "Video generation is disabled due to hardware limitations."
    elif "speak" in query.lower():
        text = query.replace("speak", "").strip()
        response = text_to_speech(text)
    elif "search" in query.lower():
        search_query = query.replace("search", "").strip()
        response = web_search(search_query)
    elif "save character" in query.lower():
        parts = query.replace("save character", "").strip().split("|")
        if len(parts) == 3:
            name, traits, appearance = parts
            save_character(name.strip(), traits.strip(), appearance.strip())
            response = f"Character '{name}' saved."
        else:
            response = "Please provide name, traits, and appearance in the format: save character name | traits | appearance"
    else:
        # General query with LLM
        full_prompt = f"{character_info}Query: {query}\nAnswer step-by-step:"
        result = llm(full_prompt, max_length=100, num_return_sequences=1)
        response = result[0]["generated_text"].split("Answer:")[-1].strip() if "Answer:" in result[0]["generated_text"] else result[0]["generated_text"]

    return response

# Streamlit UI
def main():
    st.title("AI Bot with Image Generation and Memory")
    st.write("Create images, use text-to-speech, search the web, or interact with a character-aware chatbot. Video generation is disabled due to hardware limitations.")

    # Initialize database
    init_db()

    # Input fields
    query = st.text_input("Enter your query:")
    character_name = st.text_input("Character name (optional, for memory):")
    if st.button("Submit"):
        if query:
            response = chatbot(query, character_name)
            st.write("**Response**:")
            if response.endswith(".png"):
                st.image(response, caption="Generated Image")
            elif response.endswith(".wav"):
                st.audio(response)
            else:
                st.write(response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()