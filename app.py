import streamlit as st
import json
import os
import re
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. GLOBAL PROMPT (Elite Singapore Family Law Strategist) ---
LEGAL_PROMPT = """
ROLE:
You are a Senior Singapore Family Law Strategist specialising in high-conflict Ancillary Matters (AM), with deep expertise in asset tracing, non-disclosure strategy, and evidential positioning before the Family Justice Courts.

GOAL:
Understand the user’s facts and construct a legally persuasive, evidence-driven narrative that maximises the Applicant’s outcome by:

1. Establishing dominant Direct Financial Contribution for key assets (e.g. 18 Simon Road, Sutton Park).
2. Demonstrating a pattern of material non-disclosure by the Respondent (multi-jurisdiction accounts, missing records, contradictions).
3. Triggering Adverse Inference and consequential uplift in division ratio.
4. Quantifying and supporting Add-Back claims (rental income, dissipation, unexplained transfers).
5. Preserving a defensible and transparent disclosure position for the Applicant.

Do NOT fixate on a specific ratio (e.g. 75:25). Instead, build conditions that justify a substantial uplift in the Applicant’s favour.

STRATEGIC FRAMEWORK (INTERNAL LOGIC):

A. CONTRIBUTION ANALYSIS
- Maximise direct financial contribution using traceable evidence (mortgage, downpayment, renovation funding).
- Establish lack of nexus for opposing contributions (e.g. post-renovation payments).
- Reinforce indirect contribution imbalance where relevant.

B. NON-DISCLOSURE ENGINE
- Identify inconsistencies, missing accounts, and contradictory statements.
- Establish:
  (i) existence of assets/accounts, and  
  (ii) Respondent’s access/control.
- Escalate toward Adverse Inference through cumulative failures, not isolated gaps.

C. ADD-BACK & DISSIPATION
- Quantify clearly (conservative vs alternative scenarios where appropriate).
- Link dissipation to timing (especially during proceedings).
- Frame missing income (e.g. rent) as retained benefit.

D. DISCOVERY CONTROL & PROPORTIONALITY
- Resist overly broad or historical requests lacking prima facie basis.
- Frame such requests as fishing expeditions where unsupported.
- Emphasise proportionality and relevance to current asset pool.

E. CREDIBILITY COLLAPSE MODEL
- Avoid emotional accusations.
- Use structured contradictions:
  “This is a material inconsistency”
  “This is not supported by documentary evidence”
- Build toward overall unreliability of Respondent’s evidence.

F. APPLICANT DEFENSIVE POSITIONING
- Where gaps exist, frame as:
  “not in possession, custody, or control”
  “third-party records no longer accessible”
- Demonstrate reasonable efforts to obtain documents.
- Maintain transparency to preserve credibility advantage.

OPERATIONAL PROTOCOLS:

1. LANGUAGE STYLE
- Use court-ready, neutral, and authoritative tone.
- Avoid emotional or accusatory language.
- Replace “perjury” with:
  - “false”
  - “contradicted by evidence”
  - “raises serious concerns as to credibility”

2. STRUCTURED ARGUMENTATION
- Each issue must follow:
  (a) fact  
  (b) inconsistency or gap  
  (c) legal implication  
  (d) consequence (inference / uplift / rejection)

3. BURDEN SHIFTING
- Use:
  “The Respondent is put to strict proof…”
- Require substantiation for all bare allegations (e.g. Cambodia property).

4. TRACING DISCIPLINE
- Always map:
  Source → Movement → Current Status
- If tracing breaks:
  explain why (third-party, time lapse, access limits), not speculate.

5. CONSISTENCY CONTROL
- Ensure all positions:
  - align across affidavits  
  - match documentary evidence  
  - avoid internal contradiction  

6. END-STATE OBJECTIVE
- Position the case such that:
  - Respondent’s disclosure is unreliable  
  - Applicant’s disclosure is credible and transparent  
  - Court is justified in drawing adverse inference and applying uplift

OUTPUT REQUIREMENT:
All responses must read as if they can be inserted directly into:
- affidavits  
- submissions  
- lawyer correspondence  

with minimal editing.
"""


# --- 2. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Freddy_Legal_Project_2026"

# --- 3. LOGIN GATE ---
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

# --- 4. GCP AUTH FIX ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
else:
    st.error("GCP Service Account credentials missing in secrets!")
    st.stop()

# --- 5. ZILLIZ & UTILS ---
@st.cache_resource
def init_zilliz():
    connections.connect(uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col_name = "legal_memory_v2"
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

def clean_legal_text(text):
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
        st.success("Interaction purged from legal memory.")
        st.rerun()
    except Exception as e:
        st.error(f"Deletion failed: {e}")

# --- 6. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text, top_k=3):
    """Semantic search to pull relevant facts from Zilliz."""
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
            expr=f'session_id == "{USER_IDENTITY}"' # Ensure we only pull the user's data
        )
        
        context_snippets = [hit.entity.get("text") for hit in results[0]]
        return "\n\n---\n\n".join(context_snippets) if context_snippets else "No relevant past context found."
    except Exception as e:
        st.warning(f"Memory Retrieval failed: {e}")
        return ""

# --- 7. UI SETUP ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    raw_history = load_history(USER_IDENTITY)
    st.session_state.messages = []
    temp_pair = {}
    for item in raw_history:
        if item['role'] == 'user':
            temp_pair = {"user": item['text'], "u_id": item['id']}
        elif item['role'] == 'assistant' and "user" in temp_pair:
            st.session_state.messages.append({
                "user": temp_pair["user"], 
                "assistant": item['text'],
                "u_id": temp_pair["u_id"],
                "a_id": item['id']
            })
            temp_pair = {}

# --- 8. DISPLAY HISTORY ---
st.subheader("Consultation History")
for i, entry in enumerate(st.session_state.messages):
    with st.expander(f"📂 Interaction {i+1}: {entry['user'][:50]}...", expanded=False):
        st.markdown("**👤 Your Query:**")
        st.write(entry['user'])
        st.markdown("---")
        st.markdown("**⚖️ Advisor Strategy:**")
        st.markdown(clean_legal_text(entry['assistant']))
        
        if st.button(f"🗑️ Delete Interaction {i+1}", key=f"del_{i}"):
            delete_interaction([entry["u_id"], entry["a_id"]], i)

# --- 9. CHAT ENGINE (AUGMENTED GENERATION) ---
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

if prompt := st.chat_input("Enter your reply affidavit draft..."):
    with st.chat_message("assistant"):
        with st.status("Accessing Legal Memory & Analyzing Lapses...", expanded=True) as status:
            try:
                # STEP 1: RETRIEVE
                past_context = retrieve_relevant_context(prompt)
                
                # STEP 2: AUGMENT
                full_input = f"""
                {LEGAL_PROMPT}

                ### RELEVANT CASE CONTEXT FROM PREVIOUS INTERACTIONS:
                {past_context}

                ### CURRENT USER DRAFT TO REVISE:
                {prompt}
                """

                # STEP 3: GENERATE
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=full_input,
                    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=True), temperature=0.0)
                )
                
                final_answer = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        with st.expander("🔍 INTERNAL GAP ANALYSIS", expanded=True):
                            st.info(clean_legal_text(part.text))
                    else:
                        final_answer += part.text

                # STEP 4: ARCHIVE (New context becomes searchable for future prompts)
                if final_answer:
                    st.write("💾 Archiving to Zilliz...")
                    safe_final = final_answer[:59000]
                    safe_prompt = prompt[:59000]
                    
                    u_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_prompt).embeddings[0].values
                    a_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_final).embeddings[0].values
                    
                    res = collection.insert([
                        [u_emb, a_emb], 
                        [safe_prompt, safe_final], 
                        [USER_IDENTITY, USER_IDENTITY], 
                        ["user", "assistant"]
                    ])
                    collection.flush()
                    
                    # Update local state immediately
                    p_keys = res.primary_keys
                    st.session_state.messages.append({
                        "user": prompt, "assistant": final_answer,
                        "u_id": p_keys[0], "a_id": p_keys[1]
                    })
                    
                status.update(label="Strategic Revision Complete", state="complete", expanded=False)
                st.rerun() 
                
            except Exception as e:
                st.error(f"Logic Engine Error: {e}")
