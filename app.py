import streamlit as st
import json
import os
import re
import requests
import concurrent.futures
import time
import datetime
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global"
GEMINI_MODEL = "gemini-3.1-pro-preview"
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Generic_Ensemble_User"

# OpenRouter Configuration
def get_secret(key):
    try:
        val = st.secrets[key]
        # If it's dict-like
        if hasattr(val, "get"):
            v = val.get(key, val)
            if isinstance(v, str): return v
        # If it's a string that looks like a dict
        if isinstance(val, str) and val.strip().startswith("{"):
            import ast
            parsed = ast.literal_eval(val.strip())
            if hasattr(parsed, "get"):
                return parsed.get(key, val)
        return val
    except Exception:
        return ""

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENROUTER_MODEL_A = "openai/gpt-oss-120b"

# Google AI Studio Configuration
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GEMINI_FLASH_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

# Groq Configuration
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- 2. LOGIN GATE ---
def check_password():
    if "passwords" not in st.secrets:
        st.error("🚨 Configuration Error: '[passwords]' section missing in Secrets.")
        return False
    def password_entered():
        if (st.session_state["username"] in st.secrets["passwords"] and 
            st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Log In", on_click=password_entered)
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 3. ZILLIZ & UTILS ---
@st.cache_resource
def init_zilliz():
    connections.connect(uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
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

collection = init_zilliz()

def clean_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.replace("add−back", "add-back").replace("S$", "S$ ").replace("\n", "\n\n")

def load_history(session_id):
    try:
        results = collection.query(expr=f'session_id == "{session_id}"', output_fields=["id", "text", "role"])
        return sorted(results, key=lambda x: x['id'])
    except:
        return []

def delete_interaction(ids_to_delete, index_in_state):
    try:
        delete_expr = f"id in {ids_to_delete}"
        collection.delete(delete_expr)
        collection.flush()
        st.session_state.messages.pop(index_in_state)
        st.success("Interaction purged from memory.")
        st.rerun()
    except Exception as e:
        st.error(f"Deletion failed: {e}")

# --- 5. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text, top_k=3):
    """Semantic search to pull relevant facts from Zilliz."""
    if collection.num_entities == 0:
        return ""
        
    try:
        search_emb = client.models.embed_content(
            model=EMBED_MODEL, 
            contents=query_text
        ).embeddings[0].values
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
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

def call_openrouter_a(prompt_text):
    """Call LLM A (GPT-OSS-120B) from OpenRouter."""
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API Key not configured."
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    payload = {
        "model": OPENROUTER_MODEL_A,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vmindai.streamlit.app",
        "X-Title": "VMindAI Ensemble System"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling LLM A: {e}"

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

def call_gemini_flash_synthesize(output1, output2, output3, output4):
    """Call gemini-3.1-flash-lite-preview to synthesize outputs into master output."""
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_FLASH_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    prompt_text = f"""You are an Expert Editor. Your task is to synthesize three AI responses into one master response, based on the original context and prompt.

Original Prompt & Context (Output 1):
{output1}

Response A (LLM A - GPT-OSS):
{output2}

Response B (LLM 3 - Gemini Pro):
{output3}

Response C (LLM C - Llama-4-Scout):
{output4}

Synthesize Response A, Response B, and Response C into a cohesive, comprehensive master output that:
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
st.title("🤖 Multi-LLM Ensemble System")

if "messages" not in st.session_state:
    raw_history = load_history(USER_IDENTITY)
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
            elif role in ['assistant', 'master_output']:
                current_interaction["master"] = item['text']
                current_interaction["m_id"] = item['id']
                
    if current_interaction is not None:
        st.session_state.messages.append(current_interaction)

# --- 8. DISPLAY HISTORY ---
st.subheader("Consultation History")
for i, entry in enumerate(st.session_state.messages):
    with st.expander(f"📂 Interaction {i+1}: {entry['user'][:50]}...", expanded=False):
        st.markdown("**👤 Your Query:**")
        st.write(entry['user'])
        
        if entry.get('output1'):
            st.markdown("---")
            st.markdown("**📝 Output 1: Optimized User Prompt:**")
            st.markdown(clean_text(entry['output1']))
            
        if entry.get('output2'):
            st.markdown("---")
            st.markdown("**🤖 Output 2: LLM A (GPT-OSS-120B):**")
            st.markdown(clean_text(entry['output2']))
            
        if entry.get('output3'):
            st.markdown("---")
            st.markdown("**⚡ Output 3: LLM 3 (gemini-3.1-pro-preview):**")
            st.markdown(clean_text(entry['output3']))
            
        if entry.get('output4'):
            st.markdown("---")
            st.markdown("**🚀 Output 4: LLM C (Llama-4-Scout-17B):**")
            st.markdown(clean_text(entry['output4']))
            
        if entry.get('master'):
            st.markdown("---")
            st.markdown("**🏆 Output A: Master Synthesis:**")
            st.markdown(clean_text(entry['master']))
        
        if st.button(f"🗑️ Delete Interaction {i+1}", key=f"del_{i}"):
            delete_interaction(entry["all_ids"], i)

# --- 9. CHAT ENGINE (MULTI-LLM PIPELINE) ---
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

if prompt := st.chat_input("Enter your query or draft..."):
    with st.chat_message("assistant"):
        with st.status("Processing Query...", expanded=True) as status:
            try:
                # STEP 1: RETRIEVE CONTEXT
                past_context = retrieve_relevant_context(prompt)
                status.update(label="Retrieving memory...", state="running")
                
                # STEP 2: GENERATE OUTPUT 1 (User Prompt + Context) using LLM 1 (Gemini Prompt Creator - gemini-3.1-flash-lite-preview)
                status.update(label="Generating optimized user prompt (Output 1)...", state="running")
                t0 = time.time()
                raw_output1 = call_gemini_prompt_creator(f"Context: {past_context}\n\nUser Entry: {prompt}")
                t1 = time.time() - t0
                
                # STEP 3: GENERATE OUTPUTS 2, 3, and 4 concurrently using LLM 2 (GPT-OSS-120B), LLM 3 (gemini-3.1-pro-preview), and LLM 4 (Llama-4-Scout)
                status.update(label="Generating strategies with LLMs concurrently...", state="running")
                
                def timed_call(func, arg):
                    start_t = time.time()
                    res = func(arg)
                    dur = time.time() - start_t
                    return res, dur

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_a = executor.submit(timed_call, call_openrouter_a, raw_output1)
                    future_b = executor.submit(timed_call, call_gemini_pro, raw_output1)
                    future_c = executor.submit(timed_call, call_groq_llm, raw_output1)
                    
                    raw_output2, t2 = future_a.result()
                    raw_output3, tA = future_b.result()
                    raw_output4, t4 = future_c.result()
                
                # STEP 5: SYNTHESIZE using gemini-3.1-flash-lite-preview
                status.update(label="Synthesizing master output with gemini-3.1-flash-lite-preview...", state="running")
                t0 = time.time()
                raw_master_output = call_gemini_flash_synthesize(raw_output1, raw_output2, raw_output3, raw_output4)
                t_master = time.time() - t0
                
                output1 = f"{raw_output1}\n\n*(⏱️ Time taken: {t1:.2f}s)*"
                output2 = f"{raw_output2}\n\n*(⏱️ Time taken: {t2:.2f}s)*"
                output3 = f"{raw_output3}\n\n*(⏱️ Time taken: {tA:.2f}s)*"
                output4 = f"{raw_output4}\n\n*(⏱️ Time taken: {t4:.2f}s)*"
                master_output = f"{raw_master_output}\n\n*(⏱️ Time taken: {t_master:.2f}s)*"
                
                # STEP 6: ARCHIVE ALL OUTPUTS TO ZILLIZ
                status.update(label="Archiving to Zilliz...", state="running")
                
                # Create embeddings for all outputs
                safe_prompt = prompt[:59000]
                safe_output1 = output1[:59000] if output1 else ""
                safe_output2 = output2[:59000] if output2 else ""
                safe_output3 = output3[:59000] if output3 else ""
                safe_output4 = output4[:59000] if output4 else ""
                safe_master = master_output[:59000] if master_output else ""
                
                # Generate embeddings
                try:
                    prompt_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_prompt).embeddings[0].values
                    output1_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_output1).embeddings[0].values if safe_output1 else None
                    output2_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_output2).embeddings[0].values if safe_output2 else None
                    output3_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_output3).embeddings[0].values if safe_output3 else None
                    output4_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_output4).embeddings[0].values if safe_output4 else None
                    master_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_master).embeddings[0].values if safe_master else None
                except:
                    prompt_emb = [0.0] * 768
                    output1_emb = [0.0] * 768 if safe_output1 else None
                    output2_emb = [0.0] * 768 if safe_output2 else None
                    output3_emb = [0.0] * 768 if safe_output3 else None
                    output4_emb = [0.0] * 768 if safe_output4 else None
                    master_emb = [0.0] * 768 if safe_master else None
                
                # Insert all outputs into Zilliz
                insert_data = [[prompt_emb], [safe_prompt], [USER_IDENTITY], ["user_prompt"]]
                if output1_emb:
                    insert_data[0].append(output1_emb)
                    insert_data[1].append(safe_output1)
                    insert_data[2].append(USER_IDENTITY)
                    insert_data[3].append("output1_user_prompt")
                if output2_emb:
                    insert_data[0].append(output2_emb)
                    insert_data[1].append(safe_output2)
                    insert_data[2].append(USER_IDENTITY)
                    insert_data[3].append("output2_llm_a")
                if output3_emb:
                    insert_data[0].append(output3_emb)
                    insert_data[1].append(safe_output3)
                    insert_data[2].append(USER_IDENTITY)
                    insert_data[3].append("output3_llm_b")
                if output4_emb:
                    insert_data[0].append(output4_emb)
                    insert_data[1].append(safe_output4)
                    insert_data[2].append(USER_IDENTITY)
                    insert_data[3].append("output4_llm_c")
                if master_emb:
                    insert_data[0].append(master_emb)
                    insert_data[1].append(safe_master)
                    insert_data[2].append(USER_IDENTITY)
                    insert_data[3].append("master_output")
                
                res = collection.insert(insert_data)
                collection.flush()
                
                # Update local state
                p_keys = res.primary_keys
                
                idx = 1
                o1_id = o2_id = o3_id = o4_id = m_id = None
                if output1_emb:
                    o1_id = p_keys[idx]
                    idx += 1
                if output2_emb:
                    o2_id = p_keys[idx]
                    idx += 1
                if output3_emb:
                    o3_id = p_keys[idx]
                    idx += 1
                if output4_emb:
                    o4_id = p_keys[idx]
                    idx += 1
                if master_emb:
                    m_id = p_keys[idx]
                    
                st.session_state.messages.append({
                    "user": prompt, 
                    "u_id": p_keys[0],
                    "output1": output1, "o1_id": o1_id,
                    "output2": output2, "o2_id": o2_id,
                    "output3": output3, "o3_id": o3_id,
                    "output4": output4, "o4_id": o4_id,
                    "master": master_output, "m_id": m_id,
                    "all_ids": p_keys
                })
                
                status.update(label="Analysis Complete", state="complete", expanded=False)
                
                # Display outputs
                st.subheader("📊 Multi-LLM Pipeline Output")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📝 Output 1: Optimized User Prompt", expanded=True):
                        st.markdown(clean_text(output1))
                
                with col2:
                    with st.expander("🤖 Output 2: LLM A (GPT-OSS-120B)", expanded=True):
                        st.markdown(clean_text(output2))
                
                col3, col4 = st.columns(2)
                
                with col3:
                    with st.expander("⚡ Output 3: LLM 3 (gemini-3.1-pro-preview)", expanded=True):
                        st.markdown(clean_text(output3))
                
                with col4:
                    with st.expander("🚀 Output 4: LLM C (Llama-4-Scout-17B)", expanded=True):
                        st.markdown(clean_text(output4))
                        
                st.markdown("---")
                    with st.expander("🏆 Output A: Master Synthesis (gemini-3.1-flash-lite-preview)", expanded=True):
                    st.markdown(clean_text(master_output))
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
