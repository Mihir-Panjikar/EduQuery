import streamlit as st
import time
import os
import logging
from dotenv import load_dotenv
from modules.application import EduQueryCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")

MODEL_PATH = str(os.getenv("MODEL_PATH"))
DATA_FOLDER = str(os.getenv("DATA_FOLDER"))
INDICES_FOLDER = str(os.getenv("INDICES_FOLDER", "indices"))

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(INDICES_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("--- Script Rerun ---")
print(
    f"Before init check: 'messages' in st.session_state? {'messages' in st.session_state}")

if "messages" not in st.session_state:
    print("!!! Initializing st.session_state.messages !!!")
    st.session_state.messages = []

print(f"After init check: 'messages' exists? {'messages' in st.session_state}")
print(f"Current messages type: {type(st.session_state.messages)}")

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
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_app_core():
    try:
        core = EduQueryCore(MODEL_PATH, DATA_FOLDER, INDICES_FOLDER)
        core.ensure_indices_folder()
        return core
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return None


app_core = get_app_core()

if app_core:
    subjects = app_core.get_subjects()
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = subjects[0] if subjects else None
else:
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = None

with st.spinner("Loading model..."):
    if app_core:
        if not app_core.load_model():
            st.error(
                "❌ Embedding model not found! Please download and save it in 'models/'.")
            st.stop()

st.markdown('<h1 class="main-header">📚 EduQuery (AI-Powered Q&A System)</h1>',
            unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">Ask questions based on subject-specific knowledge bases!</h3>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**Model:** {os.path.basename(MODEL_PATH)}")
    st.markdown(f"**Data Folder:** {DATA_FOLDER}")

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **EduQuery** is an AI-powered Q&A system that provides answers based on 
    subject-specific knowledge bases. It uses embeddings and 
    semantic search to find the most relevant information.
    """)

    if st.button("🔄 Refresh Subjects"):
        st.cache_resource.clear()
        st.rerun()

if app_core:
    subjects = app_core.get_subjects()

if not subjects:
    st.warning(
        f"No subject folders found in the data directory: {DATA_FOLDER}")

    # Helper to create example subjects
    with st.expander("Create Example Structure"):
        st.write("Create subject folders with documents to get started:")
        st.code(f"""
        # Example directory structure:
        {DATA_FOLDER}/
        ├── Mathematics/
        │   ├── calculus.pdf
        │   └── algebra.docx
        ├── Computer_Science/
        │   ├── programming.pdf
        │   └── algorithms.pptx
        └── Physics/
            ├── mechanics.pdf
            └── thermodynamics.docx
        """)

        if st.button("Create Example Structure"):
            example_subjects = ["Mathematics", "Computer_Science", "Physics"]
            for subject in example_subjects:
                subject_path = os.path.join(DATA_FOLDER, subject)
                os.makedirs(subject_path, exist_ok=True)

                # Create a placeholder text file
                with open(os.path.join(subject_path, "placeholder.txt"), "w") as f:
                    f.write(f"This is a placeholder file for {subject}.\n")
                    f.write(
                        "Replace with actual subject documents (.pdf, .docx, .pptx).")

            st.success(
                "Example structure created! Add your documents to these folders.")
            st.rerun()

    st.stop()

if app_core:
    missing_indices = app_core.check_missing_indices()

if missing_indices:
    if len(missing_indices) == len(subjects):
        st.warning("⚠️ No knowledge bases found. Initialize them to get started:")
    else:
        st.warning(
            f"⚠️ Knowledge bases not found for: {', '.join(missing_indices)}")

    if st.button("📚 Initialize Knowledge Bases"):
        progress_bar = st.progress(0)

        def update_progress(subject, progress):
            st.info(f"Processing knowledge base for {subject}...")
            progress_bar.progress(progress)
            time.sleep(0.1)

        with st.spinner("Processing knowledge bases..."):
            if app_core:
                if app_core.initialize_knowledge_bases(update_progress):
                    st.success("✅ Knowledge bases processed successfully!")
                    missing_indices = []
                else:
                    st.error("❌ Failed to process some knowledge bases.")

col1, col2 = st.columns([1, 3])

with col1:
    selected_subject = st.selectbox("Select Subject:", subjects)

    if selected_subject in missing_indices:
        if st.button(f"Initialize {selected_subject}"):
            with st.spinner(f"Building knowledge base for {selected_subject}..."):
                if app_core:
                    if app_core.initialize_subject(selected_subject):
                        st.success(
                            f"✅ {selected_subject} knowledge base created!")
                        missing_indices.remove(selected_subject)
                    else:
                        st.error(
                            f"❌ Failed to create {selected_subject} knowledge base.")

with col2:
    query = st.text_input("Ask a question:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if selected_subject in missing_indices:
        error_message = f"Knowledge base for {selected_subject} not found. Please initialize it first."
        st.error(error_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)
    elif app_core is None:
        error_message = "Application core is not initialized properly. Please refresh the page."
        st.error(error_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)
    else:
        with st.spinner("Finding and synthesizing answer..."):
            synthesized_answer = app_core.get_answer(query, selected_subject)

        with st.chat_message("assistant"):
            st.markdown(synthesized_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": synthesized_answer})

messages = st.session_state.messages
num_messages = len(messages)

for i in range(num_messages - 1, 0, -2):
    assistant_message = messages[i]
    user_message = messages[i-1]

    if user_message["role"] == "user":
        with st.chat_message(user_message["role"]):
            st.markdown(user_message["content"])

    if assistant_message["role"] == "assistant":
        with st.chat_message(assistant_message["role"]):
            st.markdown(assistant_message["content"])


# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Mihir Panjikar")
