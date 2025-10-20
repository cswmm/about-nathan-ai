import streamlit as st
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


# ---- PAGE SETUP ----
st.set_page_config(
    page_title="About Nathan",
    page_icon="💬",
    layout="centered",
)
st.markdown("""
    <style>
        .main-title {
            font-size: 3.2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.2em;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 1.5em;
        }
    </style>
    <div class="main-title">💼 About Nathan</div>
    <div class="subtitle">Ask me anything about Nathan’s experience, skills, or projects.</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### About This Chatbot")
    st.write("This AI assistant retrieves insights from Nathan’s portfolio and projects.")
    st.write("---")
    st.markdown("**Built with:** LangChain + OpenAI + Streamlit")
    st.write("---")
    st.markdown("**Ask about my:** Experience, Projects, Skills, Clubs, Hobbies, etc")

# ---- LOAD & PREPARE DOCS ----
@st.cache_resource
def load_vectorstore():
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---- OPENAI CLIENT ----
with st.spinner("..."):
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    embeddings = OpenAIEmbeddings(api_key=st.secrets["openai_api_key"]) 
    vectorstore = load_vectorstore()

# ---- CHAT SESSION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- USER INPUT ----
if prompt := st.chat_input("Ask me about Nathan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Retrieve relevant context ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    with st.spinner("..."):
        results = retriever.invoke(prompt)
    context = "\n\n".join([doc.page_content for doc in results])

    # --- Compose system + user messages for RAG ---
    messages = [
        {
            "role": "system",
            "content": (
                "You are Nathan's personal portfolio assistant. "
                "Use the retrieved context to answer accurately. "
                "If the answer isn't in the context, say you’re not sure.\n\n"
                f"Context:\n{context}"
            ),
        }
    ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # --- Generate response with streaming ---
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})