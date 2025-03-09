import streamlit as st
st.set_page_config(page_title="Needs More Words!", page_icon="🔠")
import os
from dotenv import load_dotenv


# 1) Load environment variables
load_dotenv()


# For st_login_form, you can specify them here OR rely on environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")      # e.g. "https://<project>.supabase.co"
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_TABLE="users"


# 2) Import st_login_form
from st_login_form import login_form

# 3) Your existing utilities
from utils import perform_analysis, display_editor, display_serp_details, display_gsc_analytics

def main_app():
    """Main content after successful login."""
    # Initialize session state variables
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'step' not in st.session_state:
        st.session_state['step'] = 'analysis'

    with st.sidebar:
        with st.expander("Navigation", icon="🔍"):
            # Let users log out
            if st.button("Log Out"):
                st.session_state["authenticated"] = False
                st.session_state["username"] = None
                st.rerun()

            # We could add a sidebar button to jump to the GSC Analysis
            if st.button("Go to GSC Analysis"):
                st.session_state['step'] = 'gsc_analysis'
                st.rerun()

    st.title('Needs More Words! Optimize Your Content')

    # Standard app flow
    if st.session_state['step'] == 'analysis':
        st.write("""
        Welcome to **Needs More Words**!
        Begin by entering a keyword to retrieve and analyze content, or connect your GSC to analyze your existing content.
        """)
        keyword = st.text_input('Enter a keyword:')
        
        if st.button('Start Analysis'):
            if keyword:
                st.session_state['keyword'] = keyword
                perform_analysis(keyword)
                st.session_state['analysis_completed'] = True
                st.session_state['step'] = 'editor'
                st.success("Analysis completed! Proceeding to the Content Editor.")
                st.rerun()
            else:
                st.error('Please enter a keyword.')

    elif st.session_state['step'] == 'editor':
        if not st.session_state['analysis_completed']:
            st.warning("Please perform the analysis first.")
            st.session_state['step'] = 'analysis'
            st.rerun()
        else:
            st.header('✍️ Content Editor')
            display_editor()

    elif st.session_state['step'] == 'serp_details':
        # Our new step that shows detailed SERP analysis
        display_serp_details()

    elif st.session_state['step'] == 'gsc_analysis':
        # <--- NEW step for GSC analysis
        display_gsc_analytics()

    else:
        # Fallback to default
        st.session_state['step'] = 'analysis'
        st.rerun()

def main():
    """
    Entry point: Show the st_login_form if not authenticated.
    Once authenticated, hide the login form and only show the welcome 
    message once. Then display the main_app.
    """
    # 1) Initialize session keys if needed
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None

    # 2) Only show the login form if not authenticated
    if not st.session_state["authenticated"]:
        # Show the login form
        client = login_form()

        # If login_form sets authenticated = True, immediately rerun
        if st.session_state["authenticated"]:
            st.rerun()
    else:
        # 3) Show the welcome message only once
        if "welcome_shown" not in st.session_state:
            if st.session_state["username"]:
                st.success(f"Welcome {st.session_state['username']}!")
                user_email = st.session_state["username"]
            else:
                st.success("Welcome, guest!")
                user_email = "guest"
            # Mark that we have shown the welcome message
            st.session_state["welcome_shown"] = True

        # 4) Then show the main app
        main_app()

# Run it!
if __name__ == "__main__":
    # If no session keys exist yet, init them
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
    main()
