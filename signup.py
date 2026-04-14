import streamlit as st
from Storage import init_auth_db, encrypt_data

@st.dialog("Create Account")
def show_signup_dialog():
    st.markdown("### Create Account")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    new_zilliz_uri = st.text_input("Zilliz Cloud URI", placeholder="https://xxx.cloud.zilliz.com")
    new_zilliz_token = st.text_input("Zilliz Token", type="password")
    
    if st.button("Create Account", use_container_width=True):
        if not (new_username and new_password and new_zilliz_uri and new_zilliz_token):
            st.error("Please fill in all fields")
            return
            
        if new_password != confirm_password:
            st.error("Passwords do not match")
            return
            
        try:
            # Use credentials from st.secrets for the auth DB itself
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
                
                auth_col.insert([
                    [dummy_vec],
                    [new_username],
                    [enc_pwd],
                    [enc_token],
                    [enc_uri]
                ])
                auth_col.flush()
                
                st.success("Account created successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Signup error: {e}")
