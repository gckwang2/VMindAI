import streamlit as st
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

def _get_encryption_key():
    """Helper to get ENCRYPTION_KEY from nested or flat secrets format."""
    secret = st.secrets.get("ENCRYPTION_KEY")
    
    # If it is a dictionary, try to get the inner key
    if isinstance(secret, dict):
        val = secret.get("ENCRYPTION_KEY")
        if val:
            st.success("DEBUG: ENCRYPTION_KEY successfully retrieved from nested section")
            return val
        else:
            st.error(f"DEBUG: ENCRYPTION_KEY section found, but inner key missing. Contents: {secret}")
            return ""
            
    # If secret is a string, return it directly
    if isinstance(secret, str):
        st.success("DEBUG: ENCRYPTION_KEY successfully retrieved (flat format)")
        return secret
        
    st.error(f"DEBUG: ENCRYPTION_KEY is neither string nor dict. Type: {type(secret)}, Value: {secret}")
    return ""

def encrypt_data(data):
    from cryptography.fernet import Fernet
    key = _get_encryption_key()
    if not key:
        raise ValueError("ENCRYPTION_KEY not found in secrets")
    
    # Ensure key is a string and valid for Fernet
    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        # If it's something else, try converting to string first
        key_bytes = str(key).encode('utf-8')

    try:
        cipher = Fernet(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Fernet key: {e}. Key must be 32-byte base64-encoded. Key provided: {key}")
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(data):
    from cryptography.fernet import Fernet
    key = _get_encryption_key()
    if not key:
        raise ValueError("ENCRYPTION_KEY not found in secrets")
    
    # Ensure key is a string and valid for Fernet
    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        key_bytes = str(key).encode('utf-8')
        
    try:
        cipher = Fernet(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Fernet key: {e}. Key must be 32-byte base64-encoded. Key provided: {key}")
    return cipher.decrypt(data.encode()).decode()

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
        key = _get_encryption_key()
        if not key:
            return None, None
        cipher = Fernet(key.encode())
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

def init_auth_db(uri, token):
    """Initializes/Returns the authentication collection."""
    connections.connect(uri=uri, token=token)
    col_name = "user_credentials_v1"
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="username", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="encrypted_password", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="encrypted_zilliz_token", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="zilliz_uri", dtype=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields)
        col = Collection(col_name, schema)
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col
