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
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. CONFIG & IDENTITY ---
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Generic_Ensemble_User"

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

# Encryption Setup
ENCRYPTION_KEY = get_secret("ENCRYPTION_KEY")
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(data):
    return cipher_suite.decrypt(data.encode()).decode()

# DashScope Configuration
DASHSCOPE_API_KEY = get_secret("DASHSCOPE_API_KEY")
DASHSCOPE_MODEL = "qwen3.5-122b-a10b"

# Google AI Studio Configuration
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GEMINI_FLASH_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

# Groq Configuration
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- 2. ZILLIZ & UTILS ---

@st.cache_resource
def init_zilliz_v2(uri, token):
    connections.connect(uri=uri, token=token)
    col_name = "ensemble_memory_v1"
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), 
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=60000), 
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=20)
        ]
        schema = CollectionSchema(fields)
        col = Collection(col_name, schema)
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col

# ... (other functions)

def populate_test_user():
    # Only populate if we have a logged-in user or session context with credentials
    if not st.session_state.get("user_zilliz_uri"):
        return
    
    auth_col = init_auth_db(
        decrypt_data(st.session_state["user_zilliz_uri"]),
        decrypt_data(st.session_state["user_zilliz_token"])
    )
    # ... rest of population logic

# Run population on startup
# populate_test_user()  # Disabled since we can't guarantee secrets are present

def get_active_collection():
    if st.session_state.get("logged_in") and st.session_state.get("user_zilliz_uri"):
        try:
            uri = decrypt_data(st.session_state["user_zilliz_uri"])
            token = decrypt_data(st.session_state["user_zilliz_token"])
            return init_zilliz_v2(uri, token)
        except Exception as e:
            st.error(f"Error decrypting user credentials: {e}")
    # If not logged in or no credentials, show warning and return None
    st.warning("Please log in to access your data.")
    return None

def retrieve_relevant_context(query_text, session_id, top_k=3):
    """Semantic search to pull relevant facts from Zilliz."""
    uri, token = get_active_credentials()
    col = init_zilliz_v2(uri, token)
    if not col or col.num_entities == 0:
        return ""
        
    try:
        search_emb = client.models.embed_content(
            model=EMBED_MODEL, 
            contents=query_text
        ).embeddings[0].values
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = col.search(
            data=[search_emb], 
            anns_field="vector", 
            param=search_params, 
            limit=top_k, 
            output_fields=["text"],
            expr=f'session_id == "{USER_IDENTITY}"'
        )
        
        context_snippets = [hit.entity.get("text") for hit in results[0]]
        return "\n\n---\n\n".join(context_snippets) if context_snippets else ""
    except Exception as e:
        st.warning(f"Memory Retrieval failed: {e}")
        return ""

# --- 6. LLM CALLS ---

def call_gemini_prompt_creator(prompt_text):
    """Call gemini-3.1-flash-lite-preview from Google AI Studio to create user prompt."""
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_FLASH_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""You are a prompt engineer. Your task is to:
1. Analyze the user's raw entry
2. Retrieve relevant context from the knowledge base
3. Synthesize a comprehensive, well-structured prompt (Output1) that includes:
   - User's core question/request
   - Relevant facts and context
   - Clear instructions for the LLMs

Current Date: {current_date}
IMPORTANT: Do not limit your responses or the generated prompt to the years 2023-2024. Acknowledge the current date and ensure the prompt is relevant to the present and future.

User Entry: {prompt_text}

Return ONLY the synthesized prompt (Output1), nothing else."""
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating prompt.")
    except Exception as e:
        return f"Error calling Gemini Prompt Creator: {e}"

def call_qwen(prompt_text):
    """Call LLM 2 (Qwen 3.5 122B) from DashScope."""
    if not DASHSCOPE_API_KEY:
        return "Error: DashScope API Key not configured."
    
    # Use OpenAI-compatible endpoint for DashScope
    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return f"Error calling Qwen (Status {response.status_code}): {response.text}"
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling Qwen: {e}"

def call_gemini_pro(prompt_text):
    """Call LLM 3 (gemini-3.1-pro-preview) from Google AI Studio."""
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_PRO_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided.\n\nPrompt: {prompt_text}"
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating response.")
    except Exception as e:
        return f"Error calling Gemini Pro: {e}"

def call_groq_llm(prompt_text):
    """Call Groq LLM."""
    if not GROQ_API_KEY:
        return "Error: Groq API Key not configured."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling Groq: {e}"

def call_gemini_flash_synthesize(output1, output2, output3, output4, output5):
    """Call gemini-3.1-flash-lite-preview to synthesize outputs into master output."""
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_FLASH_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    prompt_text = f"""You are an Expert Editor. Your task is to synthesize four AI responses into one master response, based on the original context and prompt.

Original Prompt & Context (Output 1):
{output1}

Response A (LLM 2 - Qwen 3.5 122B):
{output2}

Response B (LLM 3 - Gemini Pro):
{output3}

Response C (LLM 4 - Llama-4-Scout):
{output4}

Response D (LLM 5 - Meta AI Web):
{output5}

Synthesize Response A, Response B, Response C, and Response D into a cohesive, comprehensive master output that:
1. Integrates the strongest insights from all responses
2. Resolves any contradictions
3. Provides a unified, authoritative response
4. Maintains a professional tone

Return ONLY the master output, nothing else."""
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error synthesizing outputs.")
    except Exception as e:
        return f"Error calling Gemini Flash: {e}"

# --- 7. UI SETUP ---
st.set_page_config(page_title="Ensemble AI System", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with login/logout
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
        st.info("Please login to save interactions")
        if st.button("Login / Sign Up", use_container_width=True):
            # Show auth dialog inline
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
                                    st.success("Login successful!")
                                    time.sleep(1)
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
                                uri = st.session_state["user_zilliz_uri"]
                                token = st.session_state["user_zilliz_token"]
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
    
    st.markdown("---")
    st.markdown("### Cloud Storage")
    if st.button("Configure Cloud Storage", use_container_width=True):
        show_cloud_storage_dialog()
    
    st.markdown("---")
    st.markdown("**Supported LLMs:**")
    st.markdown("- Qwen 3.5 122B")
    st.markdown("- Gemini Pro")
    st.markdown("- Llama-4-Scout")
    st.markdown("- Gemini Flash Lite")

st.title("Multi-LLM Ensemble System")

# Only load history if logged in
if st.session_state.get("logged_in"):
    current_username = st.session_state["username"]
    uri, token = get_active_credentials()

    raw_history = load_history(uri, token, current_username)
    st.session_state.messages = []
    current_interaction = None
    for item in raw_history:
        role = item['role']
        if role in ['user', 'user_prompt']:
            if current_interaction is not None:
                st.session_state.messages.append(current_interaction)
            current_interaction = {
                "user": item['text'],
                "u_id": item['id'],
                "output1": "", "o1_id": None,
                "output2": "", "o2_id": None,
                "output3": "", "o3_id": None,
                "output4": "", "o4_id": None,
                "output5": "", "o5_id": None,
                "master": "", "m_id": None,
                "all_ids": [item['id']]
            }
        elif current_interaction is not None:
            current_interaction["all_ids"].append(item['id'])
            if role == 'output1_user_prompt':
                current_interaction["output1"] = item['text']
                current_interaction["o1_id"] = item['id']
            elif role == 'output2_llm_a':
                current_interaction["output2"] = item['text']
                current_interaction["o2_id"] = item['id']
            elif role == 'output3_llm_b':
                current_interaction["output3"] = item['text']
                current_interaction["o3_id"] = item['id']
            elif role == 'output4_llm_c':
                current_interaction["output4"] = item['text']
                current_interaction["o4_id"] = item['id']
            elif role == 'output5_llm_d':
                current_interaction["output5"] = item['text']
                current_interaction["o5_id"] = item['id']
            elif role in ['assistant', 'master_output']:
                current_interaction["master"] = item['text']
                current_interaction["m_id"] = item['id']

    if current_interaction is not None:
        st.session_state.messages.append(current_interaction)

    # --- 8. DISPLAY HISTORY ---
    st.subheader("Consultation History")
    for i, entry in enumerate(st.session_state.messages):
        with st.expander(f"Interaction {i+1}: {entry['user'][:50]}...", expanded=False):
            st.markdown("**Your Query:**")
            st.write(entry['user'])

            if entry.get('output1'):
                st.markdown("---")
                st.markdown("**Output 1: Optimized User Prompt:**")
                st.markdown(clean_text(entry['output1']))

            if entry.get('output2'):
                st.markdown("---")
                st.markdown("**Output 2: LLM 2 (Qwen 3.5 122B):**")
                st.markdown(clean_text(entry['output2']))

            if entry.get('output3'):
                st.markdown("---")
                st.markdown("**Output 3: LLM 3 (gemini-3.1-pro-preview):**")
                st.markdown(clean_text(entry['output3']))

            if entry.get('output4'):
                st.markdown("---")
                st.markdown("**Output 4: LLM C (Llama-4-Scout-17B):**")
                st.markdown(clean_text(entry['output4']))

            if entry.get('output5'):
                st.markdown("---")
                st.markdown("**Output 5: LLM 5 (Meta AI Web):**")
                st.markdown(clean_text(entry['output5']))

            if entry.get('master'):
                st.markdown("---")
                st.markdown("**Master Synthesis:**")
                st.markdown(clean_text(entry['master']))

            if st.button(f"Delete Interaction {i+1}", key=f"del_{i}"):
                delete_interaction_wrapper(entry["all_ids"], i)
else:
    st.warning("Please log in to view interaction history.")


# --- 9. CHAT ENGINE (MULTI-LLM PIPELINE) ---

def get_embedding(client, embed_model, text):
    if not text: return None
    try:
        return client.models.embed_content(model=embed_model, contents=text).embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return [0.0] * 768

if prompt := st.chat_input("Enter your query or draft..."):
    if not st.session_state.get("logged_in", False):
        st.session_state["pending_prompt"] = prompt
        # Show inline auth dialog using tabs
        auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])
        with auth_tab1:
            st.markdown("### Login")
            login_username = st.text_input("Username", key="login_username_inline")
            login_password = st.text_input("Password", type="password", key="login_password_inline")
            if st.button("Login", use_container_width=True, key="login_inline_btn"):
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
                                st.success("Login successful!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Invalid password")
                        else:
                            st.error("User not found")
                    except Exception as e:
                        st.error(f"Login error: {e}")
                else:
                    st.error("Please enter username and password")
        
        with auth_tab2:
            st.markdown("### Create Account")
            new_username = st.text_input("Username", key="signup_username_inline")
            new_password = st.text_input("Password", type="password", key="signup_password_inline")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_inline")
            new_zilliz_uri = st.text_input("Zilliz Cloud URI", key="signup_uri_inline", placeholder="https://xxx.cloud.zilliz.com")
            new_zilliz_token = st.text_input("Zilliz Token", type="password", key="signup_token_inline")
            if st.button("Create Account", use_container_width=True, key="signup_inline_btn"):
                if new_username and new_password and new_zilliz_uri and new_zilliz_token:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        try:
                            uri = st.session_state["user_zilliz_uri"]
                            token = st.session_state["user_zilliz_token"]
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
        st.markdown("**Supported LLMs:**")
        st.markdown("- Qwen 3.5 122B")
        st.markdown("- Gemini Pro")
        st.markdown("- Llama-4-Scout")
        st.markdown("- Gemini Flash Lite")
        st.stop()

if prompt or (st.session_state.get("pending_prompt") and st.session_state.get("logged_in", False)):
    actual_prompt = prompt if prompt else st.session_state.pop("pending_prompt")
    with st.chat_message("assistant"):
        with st.status("Processing Query...", expanded=True) as status:
            try:
                pipeline_start = time.time()
                current_username = st.session_state["username"]
                uri, token = get_active_credentials()
                col = init_zilliz(uri, token)

                status.update(label="Retrieving memory...", state="running")
                t_context = time.time() - pipeline_start
                past_context = retrieve_relevant_context(actual_prompt, USER_IDENTITY, col, client, EMBED_MODEL)
                status.update(label="Generating optimized user prompt (Output 1)...", state="running")
                t0 = time.time()
                raw_output1 = call_gemini_prompt_creator(GOOGLE_API_KEY, GEMINI_FLASH_MODEL, f"Context: {past_context}\n\nUser Entry: {actual_prompt}")
                t1 = time.time() - t0

                status.update(label="Generating strategies with LLMs concurrently...", state="running")
                def timed_call(func, *args):
                    start_t = time.time()
                    res = func(*args)
                    dur = time.time() - start_t
                    return res, dur

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_a = executor.submit(timed_call, call_qwen, DASHSCOPE_API_KEY, DASHSCOPE_MODEL, raw_output1)
                    future_b = executor.submit(timed_call, call_gemini_pro, GOOGLE_API_KEY, GEMINI_PRO_MODEL, raw_output1)
                    future_c = executor.submit(timed_call, call_groq_llm, GROQ_API_KEY, GROQ_MODEL, raw_output1)
                    
                    raw_output2, t2 = future_a.result()
                    raw_output3, tA = future_b.result()
                    raw_output4, t4 = future_c.result()
                    
                    raw_output5, t5 = "LLM 5 Disabled for testing.", 0.0

                status.update(label="Synthesizing master output...", state="running")
                t0 = time.time()
                raw_master_output = call_gemini_flash_synthesize(GOOGLE_API_KEY, GEMINI_FLASH_MODEL, raw_output1, raw_output2, raw_output3, raw_output4, raw_output5)
                t_master = time.time() - t0

                output1 = f"{raw_output1}\n\n*(Time taken: {t1:.2f}s)*"
                output2 = f"{raw_output2}\n\n*(Time taken: {t2:.2f}s)*"
                output3 = f"{raw_output3}\n\n*(Time taken: {tA:.2f}s)*"
                output4 = f"{raw_output4}\n\n*(Time taken: {t4:.2f}s)*"
                output5 = f"{raw_output5}\n\n*(Time taken: {t5:.2f}s)*"
                master_output = f"{raw_master_output}\n\n*(Time taken: {t_master:.2f}s)*"

                status.update(label="Archiving to Zilliz (Generating embeddings)...", state="running")
                t_archive_start = time.time()

                safe_prompt = actual_prompt[:20000]
                safe_output1 = output1[:20000] if output1 else ""
                safe_output2 = output2[:20000] if output2 else ""
                safe_output3 = output3[:20000] if output3 else ""
                safe_output4 = output4[:20000] if output4 else ""
                safe_output5 = output5[:20000] if output5 else ""
                safe_master = master_output[:20000] if master_output else ""

                prompt_emb = get_embedding(client, EMBED_MODEL, safe_prompt)
                output1_emb = get_embedding(client, EMBED_MODEL, safe_output1)
                output2_emb = get_embedding(client, EMBED_MODEL, safe_output2)
                output3_emb = get_embedding(client, EMBED_MODEL, safe_output3)
                output4_emb = get_embedding(client, EMBED_MODEL, safe_output4)
                output5_emb = get_embedding(client, EMBED_MODEL, safe_output5)
                master_emb = get_embedding(client, EMBED_MODEL, safe_master)

                current_username = st.session_state["username"]
                insert_data = [[prompt_emb], [safe_prompt], [current_username], ["user_prompt"]]
                if output1_emb:
                    insert_data[0].append(output1_emb); insert_data[1].append(safe_output1); insert_data[2].append(current_username); insert_data[3].append("output1_user_prompt")
                if output2_emb:
                    insert_data[0].append(output2_emb); insert_data[1].append(safe_output2); insert_data[2].append(current_username); insert_data[3].append("output2_llm_a")
                if output3_emb:
                    insert_data[0].append(output3_emb); insert_data[1].append(safe_output3); insert_data[2].append(current_username); insert_data[3].append("output3_llm_b")
                if output4_emb:
                    insert_data[0].append(output4_emb); insert_data[1].append(safe_output4); insert_data[2].append(current_username); insert_data[3].append("output4_llm_c")
                if output5_emb:
                    insert_data[0].append(output5_emb); insert_data[1].append(safe_output5); insert_data[2].append(current_username); insert_data[3].append("output5_llm_d")
                if master_emb:
                    insert_data[0].append(master_emb); insert_data[1].append(safe_master); insert_data[2].append(current_username); insert_data[3].append("master_output")

                res = store_interaction(uri, token, insert_data)

                t_archive = time.time() - t_archive_start
                pipeline_duration = time.time() - pipeline_start

                master_output = f"{master_output}\n\n*(Total Pipeline Time: {pipeline_duration:.2f}s | Context Retrieval: {t_context:.2f}s | Archiving & Embeddings: {t_archive:.2f}s)*"

                p_keys = res.primary_keys

                for idx, key in enumerate(p_keys[1:], 1):
                    st.session_state.messages.append({
                        "user": actual_prompt if idx == 1 else "",
                        "u_id": p_keys[0],
                        "output1": output1 if idx == 1 else "",
                        "o1_id": key if idx == 1 else None,
                        "output2": output2 if idx == 2 else "",
                        "o2_id": key if idx == 2 else None,
                        "output3": output3 if idx == 3 else "",
                        "o3_id": key if idx == 3 else None,
                        "output4": output4 if idx == 4 else "",
                        "o4_id": key if idx == 4 else None,
                        "output5": output5 if idx == 5 else "",
                        "o5_id": key if idx == 5 else None,
                        "master": master_output if idx == 6 else "",
                        "m_id": key if idx == 6 else None,
                        "all_ids": p_keys[:idx+1]
                    })

                st.session_state["logged_in"] = True
                status.update(label="Analysis Complete", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

    elif st.session_state.get("pending_prompt") and st.session_state.get("logged_in", False):
        # Re-run
        pass
