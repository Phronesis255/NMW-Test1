import streamlit as st

st.set_page_config(page_title="Needs More Words!", page_icon="🔠")

from utils import *

# Initialize session state variables
if 'analysis_completed' not in st.session_state:
    st.session_state['analysis_completed'] = False
if 'step' not in st.session_state:
    st.session_state['step'] = 'analysis'

# Title and description
st.title('Needs More Words! Optimize Your Content')

# Control flow based on the current step
if st.session_state['step'] == 'analysis':
    st.write("""
    Welcome to the **Needs More Words** app! Begin by entering a keyword to retrieve and analyze content.
    """)
    # Keyword input and Start Analysis button
    keyword = st.text_input('Enter a keyword to retrieve content:')
    start_analysis = st.button('Start Analysis')

    if start_analysis:
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


else:
    # Default to analysis step if step is undefined
    st.session_state['step'] = 'analysis'
    st.rerun()
