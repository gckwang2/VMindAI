import streamlit as st
import toml
import os
from google import genai

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        
    google_key = secrets.get("GOOGLE_API_KEY")
    if isinstance(google_key, dict):
        google_key = google_key.get("GOOGLE_API_KEY")
        
    print(f"Key loaded: {bool(google_key)}")
    
    client = genai.Client(api_key=google_key)
    res = client.models.embed_content(model="text-embedding-004", contents="Hello world")
    
    emb = res.embeddings[0].values
    print(f"Type: {type(emb)}")
    if emb:
        print("Emb is truthy")
    else:
        print("Emb is falsy")
except Exception as e:
    print(f"Error: {e}")
