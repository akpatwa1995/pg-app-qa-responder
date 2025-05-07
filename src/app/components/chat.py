"""Chat interface component for the Streamlit app."""
import streamlit as st
from typing import List, Dict

def init_chat_state():
    """Initialize chat state if not exists."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_interface(messages: List[Dict]):
    """Render the chat interface with message history."""
    # Create a container with custom CSS for scrolling
    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            position: relative;
        }
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            margin-bottom: 60px;  /* Space for input box */
        }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background: white;
            border-top: 1px solid #e0e0e0;
            z-index: 1000;
        }
        .stChatMessage {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create main chat container
    with st.container():
        # Messages container with scrolling
        with st.container():
            for message in messages:
                avatar = "🤖" if message["role"] == "assistant" else "😎"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content}) 