import streamlit as st
from streamlit_oauth import OAuth2Component
import os
import base64
import json

# import logging
# logging.basicConfig(level=logging.INFO)

st.title("Google OIDC Example")
st.write("This example shows how to use the raw OAuth2 component to authenticate with a Google OAuth2 and get email from id_token.")

# create an OAuth2Component instance
CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"


if "auth" not in st.session_state:
    # create a button to start the OAuth2 flow
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    result = oauth2.authorize_button(
        name="Continue with Google",
        icon="https://www.google.com.tw/favicon.ico",
        redirect_uri="https://needsmorewords.streamlit.app/component/streamlit_oauth.authorize_button",
        scope="openid email profile", #https://www.googleapis.com/auth/webmasters.readonly	
        key="google",
        extras_params={"prompt": "consent", "access_type": "offline"},
        use_container_width=True,
        pkce='S256',
    )
    
    if result:
        st.write(result)
        # decode the id_token jwt and get the user's email address
        id_token = result["token"]["id_token"]
        # verify the signature is an optional step for security
        payload = id_token.split(".")[1]
        # add padding to the payload if needed
        payload += "=" * (-len(payload) % 4)
        payload = json.loads(base64.b64decode(payload))
        email = payload["email"]
        st.session_state["auth"] = email
        st.session_state["token"] = result["token"]
        st.rerun()
else:
    st.write("You are logged in!")
    st.write(st.session_state["auth"])
    st.write(st.session_state["token"])
    if st.button("Logout"):
        del st.session_state["auth"]
        del st.session_state["token"]