import time
import concurrent.futures
from LLMLogic import (
    call_gemini_prompt_creator,
    call_qwen,
    call_gemini_pro,
    call_groq_llm,
    call_gemini_flash_synthesize
)
from Storage import (
    store_interaction,
    load_history,
    delete_interaction,
    get_active_credentials,
    init_zilliz
)
from google import genai
from google.genai import types
import streamlit as st
from Crypto.Util.Padding import pad, unpad
import base64
import json

def run_chat_engine():
    # Retrieve secrets from st.session_state (populated by app.py)
    GOOGLE_API_KEY = st.session_state.get("GOOGLE_API_KEY")
    GEMINI_FLASH_MODEL = st.session_state.get("GEMINI_FLASH_MODEL")
    DASHSCOPE_API_KEY = st.session_state.get("DASHSCOPE_API_KEY")
    DASHSCOPE_MODEL = st.session_state.get("DASHSCOPE_MODEL")
    GEMINI_PRO_MODEL = st.session_state.get("GEMINI_PRO_MODEL")
    GROQ_API_KEY = st.session_state.get("GROQ_API_KEY")
    GROQ_MODEL = st.session_state.get("GROQ_MODEL")
    EMBED_MODEL = st.session_state.get("EMBED_MODEL")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.pending_prompt = None

    # Show chat input for everyone
    if prompt := st.chat_input("Enter your query or draft...", key="main_chat_input"):
        # Check if logged in
        if not st.session_state.get("logged_in", False):
            # Store prompt and trigger auth dialog
            st.session_state.pending_prompt = prompt
            st.session_state.show_auth_dialog = True
            st.rerun()
        else:
            # Process the prompt if logged in
            _process_prompt(prompt, client, GOOGLE_API_KEY, GEMINI_FLASH_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_MODEL, GEMINI_PRO_MODEL, GROQ_API_KEY, GROQ_MODEL, EMBED_MODEL)

    # Check if we have a pending prompt from before login (after successful login)
    if st.session_state.get("logged_in") and st.session_state.get("pending_prompt"):
        actual_prompt = st.session_state.pop("pending_prompt")
        _process_prompt(actual_prompt, client, GOOGLE_API_KEY, GEMINI_FLASH_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_MODEL, GEMINI_PRO_MODEL, GROQ_API_KEY, GROQ_MODEL, EMBED_MODEL)

def _process_prompt(actual_prompt, client, GOOGLE_API_KEY, GEMINI_FLASH_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_MODEL, GEMINI_PRO_MODEL, GROQ_API_KEY, GROQ_MODEL, EMBED_MODEL):
    """Helper to process the actual prompt."""
    # Display user prompt immediately in chat
    with st.chat_message("user"):
        st.write(actual_prompt)
    
    # Create a status indicator
    with st.status("Processing Query...", expanded=True) as status:
        try:
            pipeline_start = time.time()
            current_username = st.session_state["username"]
            uri, token = get_active_credentials()
            col = init_zilliz(uri, token)

            status.update(label="Retrieving memory...", state="running")
            t_context = time.time() - pipeline_start
            past_context = retrieve_relevant_context(
                actual_prompt, current_username, col, client, EMBED_MODEL
            )

            status.update(label="Generating optimized user prompt (Output 1)...", state="running")
            t0 = time.time()
            raw_output1 = call_gemini_prompt_creator(
                GOOGLE_API_KEY,
                GEMINI_FLASH_MODEL,
                f"Context: {past_context}\n\nUser Entry: {actual_prompt}"
            )
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
            raw_master_output = call_gemini_flash_synthesize(
                GOOGLE_API_KEY,
                GEMINI_FLASH_MODEL,
                raw_output1, raw_output2, raw_output3, raw_output4, raw_output5
            )
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
            output1_emb = get_embedding(client, EMBED_MODEL, safe_output1) if safe_output1 else None
            output2_emb = get_embedding(client, EMBED_MODEL, safe_output2) if safe_output2 else None
            output3_emb = get_embedding(client, EMBED_MODEL, safe_output3) if safe_output3 else None
            output4_emb = get_embedding(client, EMBED_MODEL, safe_output4) if safe_output4 else None
            output5_emb = get_embedding(client, EMBED_MODEL, safe_output5) if safe_output5 else None
            master_emb = get_embedding(client, EMBED_MODEL, safe_master) if safe_master else None

            current_username = st.session_state["username"]
            insert_data = [[prompt_emb], [safe_prompt], [current_username], ["user_prompt"]]
            if output1_emb is not None:
                insert_data[0].append(output1_emb); insert_data[1].append(safe_output1); insert_data[2].append(current_username); insert_data[3].append("output1_user_prompt")
            if output2_emb is not None:
                insert_data[0].append(output2_emb); insert_data[1].append(safe_output2); insert_data[2].append(current_username); insert_data[3].append("output2_llm_a")
            if output3_emb is not None:
                insert_data[0].append(output3_emb); insert_data[1].append(safe_output3); insert_data[2].append(current_username); insert_data[3].append("output3_llm_b")
            if output4_emb is not None:
                insert_data[0].append(output4_emb); insert_data[1].append(safe_output4); insert_data[2].append(current_username); insert_data[3].append("output4_llm_c")
            if output5_emb is not None:
                insert_data[0].append(output5_emb); insert_data[1].append(safe_output5); insert_data[2].append(current_username); insert_data[3].append("output5_llm_d")
            if master_emb is not None:
                insert_data[0].append(master_emb); insert_data[1].append(safe_master); insert_data[2].append(current_username); insert_data[3].append("master_output")

            res = store_interaction(uri, token, insert_data)

            t_archive = time.time() - t_archive_start
            pipeline_duration = time.time() - pipeline_start

            master_output = f"{master_output}\n\n*(Total Pipeline Time: {pipeline_duration:.2f}s | Context Retrieval: {t_context:.2f}s | Archiving & Embeddings: {t_archive:.2f}s)*"

            # If store_interaction returns None (e.g., due to an error), handle it gracefully
            if res and hasattr(res, 'primary_keys'):
                p_keys = res.primary_keys
            else:
                p_keys = []
                st.warning("Interaction was not stored in Zilliz database properly.")

            # Store as a single interaction unit
            interaction_data = {
                "user": actual_prompt,
                "u_id": p_keys[0] if len(p_keys) > 0 else None,
                "output1": output1,
                "o1_id": p_keys[1] if len(p_keys) > 1 else None,
                "output2": output2,
                "o2_id": p_keys[2] if len(p_keys) > 2 else None,
                "output3": output3,
                "o3_id": p_keys[3] if len(p_keys) > 3 else None,
                "output4": output4,
                "o4_id": p_keys[4] if len(p_keys) > 4 else None,
                "output5": output5,
                "o5_id": p_keys[5] if len(p_keys) > 5 else None,
                "master": master_output,
                "m_id": p_keys[6] if len(p_keys) > 6 else None,
                "all_ids": p_keys
            }
            st.session_state.messages.append(interaction_data)

            # Display the master output in the chat interface
            with st.chat_message("assistant"):
                st.markdown("**Master Synthesis**:")
                st.markdown(master_output)

            status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Force a rerun to show the interaction in the history list with the delete button
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline Error: {e}")

def retrieve_relevant_context(query_text, session_id, col, client, embed_model, top_k=3):
    if col.num_entities == 0:
        return ""
    try:
        search_emb = client.models.embed_content(model=embed_model, contents=query_text).embeddings[0].values
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = col.search(data=[search_emb], anns_field="vector", param=search_params, limit=top_k, output_fields=["text"], expr=f'session_id == "{session_id}"')
        return "\n\n---\n\n".join([hit.entity.get("text") for hit in results[0]]) if results[0] else ""
    except Exception as e:
        st.warning(f"Memory Retrieval failed: {e}")
        return ""

def get_embedding(client, embed_model, text):
    if not text: return None
    try:
        return client.models.embed_content(model=embed_model, contents=text).embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return [0.0] * 768
