import streamlit as st
st.set_page_config(page_title="Needs More Words!", page_icon="🔠")
import os
from dotenv import load_dotenv

# 1) Load environment variables
load_dotenv()

# 3) Your existing utilities
from utils import perform_analysis, display_editor, display_serp_details

def main_app():
    """Main content after successful login."""
    # Initialize session state variables
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'step' not in st.session_state:
        st.session_state['step'] = 'analysis'

    st.title('Needs More Words! Optimize Your Content')

    # Standard app flow
    if st.session_state['step'] == 'analysis':
        st.write("""
        Welcome to the **Needs More Words** app!
        Begin by entering a keyword to retrieve and analyze content.
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

    else:
        # Fallback to default
        st.session_state['step'] = 'analysis'
        st.rerun()

def main():
    """
    Entry point: Directly show the main app.
    """
    main_app()

# Run it!
if __name__ == "__main__":
    main()
