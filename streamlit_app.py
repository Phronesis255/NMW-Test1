import streamlit as st
import os
from streamlit_oauth import OAuth2Component
import base64
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def connect_to_search_console(access_token, refresh_token, client_id, client_secret, scopes):
    try:
        creds = Credentials(token=access_token,
                            refresh_token=refresh_token,
                            token_uri='https://oauth2.googleapis.com/token',
                            client_id=client_id,
                            client_secret=client_secret,
                            scopes=scopes)

        service = build('searchconsole', 'v1', credentials=creds)
        print("Successfully connected to Google Search Console API.")
        return service

    except Exception as e:
        print(f"An error occurred during connection: {e}")
        return None

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
        scope="https://www.googleapis.com/auth/webmasters.readonly", #https://www.googleapis.com/auth/webmasters.readonly	
        key="google",
        extras_params={"prompt": "consent", "access_type": "offline"},
        use_container_width=True,
        pkce='S256',
    )
    
    if result:
        st.write(result)
        # decode the id_token jwt and get the user's email address
        access_token = result["token"]["access_token"]
        refresh_token = result["token"]["refresh_token"]
        st.session_state["token"] = result["token"]
        st.session_state["auth"] = "success"
else:
    st.write("You are logged in!")
    st.write(st.session_state["auth"])
    st.write(st.session_state["token"])
    if st.button("Logout"):
        del st.session_state["auth"]
        del st.session_state["token"]


if "token" in st.session_state:
    token=st.session_state["token"]
    access_token = token["access_token"]
    refresh_token = token["refresh_token"]
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

    search_console_service = connect_to_search_console(access_token, refresh_token, CLIENT_ID, CLIENT_SECRET, SCOPES)
    if search_console_service:
        st.success("Successfully connected to Google Search Console API!")

        # --- Example: List your websites (web properties) ---
        if st.button("List My Websites"):
            try:
                response = search_console_service.sites().list().execute()
                st.write("\nYour Search Console Websites:")
                if 'siteEntry' in response:
                    first_site = response['siteEntry'][0] # Get the first site
                    site_url = first_site['siteUrl']                    
                    for site in response['siteEntry']:
                        st.write(f"- {site['siteUrl']}")
                else:
                    site_url = None
                    st.write("No websites found in your Search Console account.")
            except Exception as e:
                st.error(f"Error listing websites: {e}")
                site_url = None
            if site_url: # Only proceed if we have a site URL
                # --- Date Range for Query Data ---
                start_date = '2025-01-01' # January 1st, 2025
                end_date = '2025-03-01'   # March 1st, 2025

                if st.button("Get Query Data for Jan 1, 2025 - Mar 1, 2025"):
                    try:
                        request = {
                            'startDate': start_date,
                            'endDate': end_date,
                            'dimensions': ['query'],
                            'searchType': 'web' # Default to web search
                        }

                        response = search_console_service.searchanalytics().query(
                            siteUrl=site_url, body=request).execute()

                        query_data = []
                        if 'rows' in response:
                            for row in response['rows']:
                                query = row['keys'][0] # Query is the first (and only) dimension
                                clicks = row['clicks']
                                impressions = row['impressions']
                                ctr = row['ctr']
                                position = row['position']
                                query_data.append({
                                    'Query': query,
                                    'Clicks': clicks,
                                    'Impressions': impressions,
                                    'CTR': ctr,
                                    'Position': position
                                })
                        else:
                            st.info("No query data found for the selected date range and website.")

                        if query_data:
                            df = pd.DataFrame(query_data)
                            st.dataframe(df) # Display as Streamlit DataFrame
                        else:
                            st.warning("No query data to display.")


                    except Exception as e:
                        st.error(f"Error fetching query data: {e}")
    else:
        st.error("Failed to connect to Google Search Console API. Cannot fetch query data.")
