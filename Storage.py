import streamlit as st
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- STORAGE FUNCTIONS ---

def init_zilliz(uri, token):
    """Initializes/Returns the memory collection."""
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

def get_active_collection(uri, token):
    """Gets collection using provided credentials."""
    return init_zilliz(uri, token)

def get_active_credentials():
    """Gets active Zilliz credentials from session state."""
    if st.session_state.get("user_zilliz_uri") and st.session_state.get("user_zilliz_token"):
        from cryptography.fernet import Fernet
        cipher = Fernet(st.secrets["ENCRYPTION_KEY"])
        uri = cipher.decrypt(st.session_state["user_zilliz_uri"].encode()).decode()
        token = cipher.decrypt(st.session_state["user_zilliz_token"].encode()).decode()
        return uri, token
    return None, None

def load_history(uri, token, session_id):
    """Retrieves interaction history for a session."""
    try:
        col = get_active_collection(uri, token)
        results = col.query(expr=f'session_id == "{session_id}"', output_fields=["id", "text", "role"])
        return sorted(results, key=lambda x: x['id'])
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []

def store_interaction(uri, token, insert_data):
    """Stores interaction data in Zilliz."""
    try:
        col = get_active_collection(uri, token)
        res = col.insert(insert_data)
        col.flush()
        return res
    except Exception as e:
        st.error(f"Error storing interaction: {e}")
        return None

def delete_interaction(uri, token, ids_to_delete):
    """Deletes interactions from Zilliz."""
    try:
        col = get_active_collection(uri, token)
        delete_expr = f"id in {ids_to_delete}"
        col.delete(delete_expr)
        col.flush()
        return True
    except Exception as e:
        st.error(f"Deletion failed: {e}")
        return False
