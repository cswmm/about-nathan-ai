# About Nathan AI

[Try it out here](https://aboutnathan.streamlit.app/)

This is a straightforward RAG AI project, currently in development, using OpenAI's GPT-3.5, LangChain, and Streamlit. This project uses a RAG (Retrieval Augmented Generation) pipeline to allow users to ask about my portfolio
using natural language.

For a local, lightweight, and fast retrieval, the FAISS (Facebook AI Similarity Search) library was used to find relevant info in the documents. In the future, I plan to integrate this pipeline into a full stack portfolio application. For that, I am looking to use ChromaDB to store the info persistently and deploy it as a cloud application.

### How to run it on your own machine
1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
