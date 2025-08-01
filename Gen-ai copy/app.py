import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# ✅ Set page config at the top
st.set_page_config(page_title="MIT-INFO", layout="wide")

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key not found! Please add it to a `.env` file.")
    st.stop()

# Set API Key
os.environ["OPENAI_API_KEY"] = api_key

# Load PDF file
pdf_path = "PEC_DL_Theroy_merged.pdf"

@st.cache_resource
def load_and_process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)

        return vector_store
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

vector_store = load_and_process_pdf(pdf_path)

# Conversation State
class ConversationState(TypedDict):
    history: List[dict]
    question: str
    context: List[Document]
    answer: str

# Define Retrieval & Generation functions
def retrieve(state: ConversationState):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: ConversationState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"][:3])
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": state["question"], "context": docs_content})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    response = llm.invoke(messages)

    return {"answer": response.content}

# Build RAG pipeline
graph_builder = StateGraph(ConversationState).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# ✅ Custom Styling for Centered Chat UI & Floating Input
st.markdown(
    """
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
            padding-bottom: 100px;
        }
        .user-message {
            background-color: #D4C1F7;
            color: black;
            padding: 12px;
            border-radius: 10px;
            max-width: 75%;
            margin-left: auto;
            text-align: right;
            margin-bottom: 10px; /* ✅ Added spacing */
        }
        .ai-message {
            background-color: #f3f3f3;
            color: black;
            padding: 12px;
            border-radius: 10px;
            max-width: 75%;
            margin-right: auto;
            text-align: left;
            margin-bottom: 20px; /* ✅ More space between responses */
        }
        .input-container {
            position: fixed;
            bottom: 20px;
            width: 100%;
            max-width: 700px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 10px;
            border-radius: 10px;
            display: flex;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .input-text {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            outline: none;
        }
        .send-button {
            background-color: #0d6efd;
            color: white;
            border: none;
            padding: 10px 15px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Centered Title
st.markdown("<h1 style='text-align: center;'>MIT-INFO AI</h1>", unsafe_allow_html=True)

# ✅ Maintain full history
if "history" not in st.session_state:
    st.session_state.history = []

# ✅ Display chat history in the middle
st.write("<div class='chat-container'>", unsafe_allow_html=True)
for chat in st.session_state.history:
    st.markdown(f"<div class='user-message'><b>👤 You:</b> {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ai-message'><b>🤖 Assistant:</b> {chat['assistant']}</div>", unsafe_allow_html=True)
st.write("</div>", unsafe_allow_html=True)

# ✅ User input box with a send button
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
user_input = st.text_input("💬 Type your message...", key="user_input", placeholder="Ask a question...", label_visibility="collapsed")

col1, col2 = st.columns([5, 1])
with col1:
    pass  # Just spacing
with col2:
    if st.button("ENTER ✅", key="send_button"):
        if user_input:
            conversation_state = {
                "history": st.session_state.history,
                "question": user_input,
                "context": [],
                "answer": ""
            }
            response = graph.invoke(conversation_state)

            # Store full chat history
            st.session_state.history.append({"user": user_input, "assistant": response["answer"]})

            # Rerun app to refresh chat
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
