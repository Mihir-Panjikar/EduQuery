import streamlit as st
import time
import os
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from modules.database_utils import (
    initialize_database, save_chat, create_conversation, save_message,
    fetch_conversations, fetch_conversation_messages, update_conversation_title,
    delete_conversation, delete_all_conversations
)
from modules.application import process_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")

# --- Database Initialization ---
initialize_database()

MODEL_PATH = str(os.getenv("MODEL_PATH"))
DATA_FOLDER = str(os.getenv("DATA_FOLDER"))
INDICES_FOLDER = str(os.getenv("INDICES_FOLDER", "indices"))

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(INDICES_FOLDER, exist_ok=True)
if MODEL_PATH and os.path.dirname(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []  # Current conversation messages

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = None

st.set_page_config(
    page_title="EduQuery - College ChatBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem !important;
    color: #1E88E5 !important;
}
.sub-header {
    font-size: 1.5rem !important;
    color: #424242 !important;
}
.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
}
.code-block {
    background-color: #f5f5f5;
    border-radius: 5px;
    padding: 1rem;
}
.chat-title-input {
    border: none;
    background: transparent;
    font-size: 1rem;
    font-weight: bold;
    width: 100%;
    padding: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)


# --- Core Application Logic Setup ---

def get_subjects(data_folder):
    try:
        return [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    except FileNotFoundError:
        return []


def check_missing_indices(subjects, indices_folder):
    missing = []
    for subject in subjects:
        index_path = os.path.join(indices_folder, subject)
        # Example check: look for a specific index file like faiss_index.idx
        if not os.path.exists(os.path.join(index_path, "faiss_index.idx")):
            missing.append(subject)
    return missing


def initialize_subject(subject, data_folder, indices_folder, model_path):
    # Placeholder: Add the actual logic to build index for a subject
    print(f"Initializing index for {subject}...")
    # Simulate success/failure
    try:
        # Call your actual index building function here
        time.sleep(2)  # Simulate work
        return True
    except Exception as e:
        logger.error(f"Failed to initialize {subject}: {e}")
        return False


def start_new_conversation(subject=None):
    """Starts a new conversation and sets it as the current conversation."""
    # Default title is based on the subject and current time
    title = f"{subject}" if subject else "New Conversation"
    title += f" - {datetime.now().strftime('%I:%M %p')}"

    # Create the conversation in the database
    conversation_id = create_conversation(title, subject)

    # Update session state
    st.session_state.current_conversation_id = conversation_id
    st.session_state.conversation_title = title
    st.session_state.messages = []  # Clear current messages
    return conversation_id


def load_conversation(conversation_id):
    """Loads an existing conversation into the session state."""
    if conversation_id:
        messages = fetch_conversation_messages(conversation_id)
        st.session_state.messages = messages
        st.session_state.current_conversation_id = conversation_id

        # Find the conversation title from the currently displayed conversations
        for conv in st.session_state.get('conversations', []):
            if conv['id'] == conversation_id:
                st.session_state.conversation_title = conv['title']
                break
    else:
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.conversation_title = None


subjects = get_subjects(DATA_FOLDER)
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = subjects[0] if subjects else None

# --- UI Rendering ---

st.markdown('<h1 class=\"main-header\">📚 EduQuery (AI-Powered Q&A System)</h1>',
            unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">Ask questions based on subject-specific knowledge bases!</h3>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")
    # Use selected_subject from session state if available
    selected_subject_sidebar = st.selectbox(
        "Select Subject:",
        subjects,
        index=subjects.index(
            st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0,
        key="sidebar_subject_selector"
    )
    # Update session state if selection changes
    if selected_subject_sidebar != st.session_state.selected_subject:
        st.session_state.selected_subject = selected_subject_sidebar
        # Don't clear messages when subject changes, that will be handled by conversation management
        st.rerun()  # Rerun to update main page context if needed

    st.markdown(
        f"**Model:** {os.path.basename(MODEL_PATH) if MODEL_PATH else 'N/A'}")
    st.markdown(f"**Data Folder:** {DATA_FOLDER}")

    # New Conversation button
    if st.button("New Conversation", key="new_conversation"):
        start_new_conversation(st.session_state.selected_subject)
        st.rerun()

    st.markdown("---")
    # --- Chat History Section ---
    st.markdown("### Chat History")

    # Fetch all conversations
    conversations = fetch_conversations()
    st.session_state.conversations = conversations

    if not conversations:
        st.write("No conversations yet. Start a new one!")
    else:
        # Add button to clear all conversations
        if st.button("Clear All History", key="clear_all_convs"):
            delete_all_conversations()
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.rerun()

        # Display conversations
        for conv in conversations:
            # Create a container for each conversation
            col1, col2 = st.columns([5, 1])

            with col1:
                # Make the title clickable to load the conversation
                if st.button(
                    f"{conv['title'][:25]}...",
                    key=f"load_{conv['id']}",
                    help=f"Click to load this conversation from {conv['created_at']}"
                ):
                    load_conversation(conv['id'])
                    st.rerun()

            with col2:
                # Add delete button for each conversation
                if st.button("🗑️", key=f"delete_{conv['id']}", help="Delete this conversation"):
                    delete_conversation(conv['id'])
                    # If we deleted the current conversation, clear the current state
                    if st.session_state.current_conversation_id == conv['id']:
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                    st.rerun()

    # --- About Section (at the end) ---
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **EduQuery** is an AI-powered Q&A system that provides answers based on 
    subject-specific knowledge bases. It uses embeddings and 
    semantic search to find the most relevant information.
    """)

    st.markdown("---")
    if st.button("Refresh Subjects & Check Indices"):
        st.rerun()


# --- Main Chat Interface ---

# Show current conversation title or a placeholder
if st.session_state.current_conversation_id and st.session_state.conversation_title:
    # Allow editing the conversation title
    new_title = st.text_input(
        "Conversation Title",
        value=st.session_state.conversation_title,
        key="edit_conversation_title"
    )
    if new_title != st.session_state.conversation_title and st.session_state.current_conversation_id and new_title is not None:
        # Update the title in the database - ensure we have a valid conversation ID and title
        update_conversation_title(
            str(st.session_state.current_conversation_id), str(new_title))
        st.session_state.conversation_title = new_title

# Display messages in the current conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle missing indices check
missing_indices = check_missing_indices(subjects, INDICES_FOLDER)

if st.session_state.selected_subject in missing_indices:
    st.warning(
        f"Knowledge base for {st.session_state.selected_subject} not found.")
    if st.button(f"Initialize {st.session_state.selected_subject} Knowledge Base"):
        with st.spinner(f"Building knowledge base for {st.session_state.selected_subject}..."):
            # Pass necessary paths to the initialization function
            if initialize_subject(st.session_state.selected_subject, DATA_FOLDER, INDICES_FOLDER, MODEL_PATH):
                st.success(
                    f"✅ {st.session_state.selected_subject} knowledge base created!")
                st.rerun()  # Rerun to update UI state
            else:
                st.error(
                    f"❌ Failed to create {st.session_state.selected_subject} knowledge base.")

# Chat input
if prompt := st.chat_input(f"Ask about {st.session_state.selected_subject}..."):
    if not st.session_state.selected_subject:
        st.error("Please select a subject first.")
    elif st.session_state.selected_subject in missing_indices:
        st.error(
            f"Knowledge base for {st.session_state.selected_subject} is not initialized. Please initialize it first using the button above.")
    else:
        # If no conversation is active, create a new one
        if not st.session_state.current_conversation_id:
            start_new_conversation(st.session_state.selected_subject)

        # Append user message to session state
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)

        # Save to database
        if st.session_state.current_conversation_id:
            save_message(
                str(st.session_state.current_conversation_id), 'user', prompt)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and get response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = None
            with st.spinner("Thinking..."):
                # Get response from the model
                response_data = process_query(
                    prompt, st.session_state.selected_subject)

                if isinstance(response_data, dict):
                    full_response = response_data.get(
                        'answer', "Sorry, I encountered an issue.")
                    sources = response_data.get('sources')
                else:  # Fallback if response is not a dict
                    full_response = "Sorry, received an unexpected response format."
                    sources = None

                # Save assistant message to database
                if st.session_state.current_conversation_id:
                    save_message(
                        str(st.session_state.current_conversation_id),
                        'assistant',
                        full_response,
                        sources
                    )

                # Display the response
                message_placeholder.markdown(full_response)

        # Append assistant message to session state
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat(),
            "context_sources": sources
        }
        st.session_state.messages.append(assistant_message)


# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Mihir Panjikar")
