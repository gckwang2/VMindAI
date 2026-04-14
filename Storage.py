import streamlit as st
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

def _get_encryption_key():
    """Helper to get ENCRYPTION_KEY from nested or flat secrets format."""
    # Assuming user defined [ENCRYPTION_KEY] in secrets.toml
    # Accessing it directly might return the dict or the value.
    # Let's inspect what st.secrets["ENCRYPTION_KEY"] returns.
    secret = st.secrets.get("ENCRYPTION_KEY")
    if isinstance(secret, dict):
        # If it's a section [ENCRYPTION_KEY], it contains key/value pairs
        # The user said:
        # [ENCRYPTION_KEY]
        # ENCRYPTION_KEY = "-YN34Dk30pjDzL0="
        return secret.get("ENCRYPTION_KEY", "")
    return secret

def encrypt_data(data):
    from cryptography.fernet import Fernet
    key = _get_encryption_key()
    if not key:
        raise ValueError("ENCRYPTION_KEY not found in secrets")
    # key needs to be 32 bytes base64 encoded.
    # The provided key "-YN34Dk30pjDzL0=" looks suspicious. 
    # Fernet keys MUST be 32 URL-safe base64-encoded bytes.
    # I'll try to use it as is, but if it fails, the key itself is wrong.
    try:
        cipher = Fernet(key.encode())
    except Exception as e:
        raise ValueError(f"Invalid Fernet key: {e}. Key must be 32-byte base64-encoded.")
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(data):
    from cryptography.fernet import Fernet
    key = _get_encryption_key()
    if not key:
        raise ValueError("ENCRYPTION_KEY not found in secrets")
    try:
        cipher = Fernet(key.encode())
    except Exception as e:
        raise ValueError(f"Invalid Fernet key: {e}. Key must be 32-byte base64-encoded.")
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
