import streamlit as st
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


# ---- PAGE SETUP ----
st.title("AboutNathan")
st.write("Ask me anything about Nathan! This chatbot retrieves facts from his portfolio files before answering.")

# ---- OPENAI CLIENT ----
client = OpenAI(api_key=st.secrets["openai_api_key"])
embeddings = OpenAIEmbeddings(api_key=st.secrets["openai_api_key"])

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
