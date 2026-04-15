import streamlit as st
import json
import os
import re
import requests
import concurrent.futures
import time
import datetime
from cryptography.fernet import Fernet
from MetaLlama4 import call_meta_ai
from LLMLogic import (
    call_gemini_prompt_creator,
    call_qwen,
    call_gemini_pro,
    call_groq_llm,
    call_gemini_flash_synthesize
)
from Storage import (
    load_history,
    store_interaction,
    delete_interaction,
    get_active_credentials,
    init_zilliz,
    init_auth_db,
    encrypt_data,
    decrypt_data
)
from google import genai
from google.genai import types
from ChatMain import run_chat_engine

# --- 1. CONFIG & IDENTITY ---
# Attempt to use a common embedding model
EMBED_MODEL = "text-embedding-001"
current_username = "Generic_Ensemble_User"

def get_secret(key):
    try:
        val = st.secrets[key]
        if hasattr(val, "get"):
            v = val.get(key, val)
            if isinstance(v, str): return v
        if isinstance(val, str) and val.strip().startswith("{"):
            import ast
            parsed = ast.literal_eval(val.strip())
            if hasattr(parsed, "get"):
                return parsed.get(key, val)
        return val
    except Exception:
        return ""

# API Configurations
DASHSCOPE_API_KEY = get_secret("DASHSCOPE_API_KEY")
DASHSCOPE_MODEL = "qwen3.5-122b-a10b"

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GEMINI_FLASH_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Store in session state for access in other modules
st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
st.session_state["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
st.session_state["DASHSCOPE_MODEL"] = DASHSCOPE_MODEL
st.session_state["GEMINI_FLASH_MODEL"] = GEMINI_FLASH_MODEL
st.session_state["GEMINI_PRO_MODEL"] = GEMINI_PRO_MODEL
st.session_state["GROQ_API_KEY"] = GROQ_API_KEY
st.session_state["GROQ_MODEL"] = GROQ_MODEL
st.session_state["EMBED_MODEL"] = "text-embedding-004"

# Google GenAI Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- 3. UTILITY FUNCTIONS ---

def clean_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.replace("add−back", "add-back").replace("S$", "S$ ").replace("\n", "\n\n")

def delete_interaction_wrapper(ids_to_delete, index_in_state):
    uri, token = get_active_credentials()
    if delete_interaction(uri, token, ids_to_delete):
        st.session_state.messages.pop(index_in_state)
        st.success("Interaction purged from memory.")
        st.rerun()
    else:
        st.error("Deletion failed.")

# --- 4. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text, session_id, col, client, embed_model, top_k=3):
    """Semantic search to pull relevant facts from Zilliz."""
    if not col or col.num_entities == 0:
        return ""

    try:
        search_emb = client.models.embed_content(
            model=embed_model, 
            contents=query_text
        ).embeddings[0].values

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = col.search(
            data=[search_emb], 
            anns_field="vector", 
            param=search_params, 
            limit=top_k, 
            output_fields=["text"],
            expr=f'session_id == "{session_id}"'
        )

        context_snippets = [hit.entity.get("text") for hit in results[0]]
        return "\n\n---\n\n".join(context_snippets) if context_snippets else ""
    except Exception as e:
        st.warning(f"Memory Retrieval failed: {e}")
        return ""

# --- 6. AUTHENTICATION DIALOGS ---

def show_cloud_storage_dialog():
    """Show cloud storage signup dialog."""
    @st.dialog("Cloud Storage Setup")
    def dialog_content():
        st.markdown("### Zilliz Cloud Storage")
        st.markdown("Store your prompts and AI responses securely in the cloud.")
        st.markdown("**Features:**")
        st.markdown("- 5GB free storage")
        st.markdown("- Encrypted data storage")
        st.markdown("- Access from anywhere")
        st.markdown("")
        st.markdown("Sign up now to get started:")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Sign Up for Free", use_container_width=True):
                st.markdown("[Click here to sign up](https://cloud.zilliz.com/signup)")
        
        st.warning("Note: Free tier includes 5GB storage. Additional storage requires upgrade.")
        
        if st.button("Close", use_container_width=True):
            st.rerun()
    
    dialog_content()

def show_auth_dialog():
    """Show login/signup dialog."""
    @st.dialog("Authentication Required")
    def dialog_content():
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.markdown("### Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
if st.button("Login", use_container_width=True, key="login_btn"):
            if login_username and login_password:
                try:
                    uri = st.secrets["ZILLIZ_URI"]
                    token = st.secrets["ZILLIZ_TOKEN"]
                    auth_col = init_auth_db(uri, token)
                    results = auth_col.query(
                        expr=f'username == "{login_username}"',
                        output_fields=["encrypted_password", "encrypted_zilliz_token", "zilliz_uri"]
                    )
                    if results:
                        stored_enc_pwd = results[0]["encrypted_password"]
                        decrypted_pwd = decrypt_data(stored_enc_pwd)
                        if decrypted_pwd == login_password:
                            st.session_state["logged_in"] = True
                            st.session_state["user_zilliz_uri"] = results[0]["zilliz_uri"]
                            st.session_state["user_zilliz_token"] = results[0]["encrypted_zilliz_token"]
                            st.session_state["username"] = login_username
                            st.rerun()
                        else:
                            st.error("Invalid password")
                    else:
                        st.error("User not found")
                except Exception as e:
                    st.error(f"Login error: {e}")
                else:
                    st.error("Please enter username and password")
        
        with tab2:
            st.markdown("### Create Account")
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            new_zilliz_uri = st.text_input("Zilliz Cloud URI", key="signup_uri", placeholder="https://xxx.cloud.zilliz.com")
            new_zilliz_token = st.text_input("Zilliz Token", type="password", key="signup_token")
            
            if st.button("Create Account", use_container_width=True, key="signup_btn"):
                if new_username and new_password and new_zilliz_uri and new_zilliz_token:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        try:
                            # Use master credentials from secrets to initialize auth DB
                            uri = st.secrets["ZILLIZ_URI"]
                            token = st.secrets["ZILLIZ_TOKEN"]
                            auth_col = init_auth_db(uri, token)
                            
                            existing = auth_col.query(
                                expr=f'username == "{new_username}"',
                                output_fields=["username"]
                            )
                            
                            if existing:
                                st.error("Username already exists")
                            else:
                                enc_pwd = encrypt_data(new_password)
                                enc_token = encrypt_data(new_zilliz_token)
                                enc_uri = encrypt_data(new_zilliz_uri)
                                dummy_vec = [0.0] * 128
                                
                                insert_data = [
                                    [dummy_vec],
                                    [new_username],
                                    [enc_pwd],
                                    [enc_token],
                                    [enc_uri]
                                ]
                                auth_col.insert(insert_data)
                                auth_col.flush()
                                
                                st.success("Account created! Please login.")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Signup error: {e}")
                else:
                    st.error("Please fill in all fields")
            
            st.markdown("---")
            st.markdown("Don't have a Zilliz account?")
            if st.button("Sign up for Cloud Storage", use_container_width=True):
                st.markdown("[Sign up here](https://cloud.zilliz.com/signup)")
    
    dialog_content()

# --- 7. UI SETUP ---
st.set_page_config(page_title="Ensemble AI System", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_zilliz_uri" not in st.session_state:
    st.session_state.user_zilliz_uri = ""
if "user_zilliz_token" not in st.session_state:
    st.session_state.user_zilliz_token = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar
with st.sidebar:
    st.title("Ensemble AI System")
    if st.session_state.get("logged_in"):
        st.success(f"Logged in as: {st.session_state.get('username', 'User')}")
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("user_zilliz_uri", None)
            st.session_state.pop("user_zilliz_token", None)
            st.session_state.pop("username", None)
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Please login")
        if st.button("Login / Sign Up", use_container_width=True):
            show_auth_dialog()
            
    st.markdown("---")
    if st.button("Configure Cloud Storage", use_container_width=True):
        show_cloud_storage_dialog()

st.title("Multi-LLM Ensemble System")

# History & Chat Engine
if st.session_state.get("logged_in"):
    # Load history
    current_username = st.session_state["username"]
    uri, token = get_active_credentials()
    raw_history = load_history(uri, token, current_username)

    st.subheader("Consultation History")

# Check if we need to show auth dialog (user tried to chat without logging in)
if st.session_state.get("show_auth_dialog", False):
    st.session_state.pop("show_auth_dialog", None)
    show_auth_dialog()

# Chat input
run_chat_engine()

