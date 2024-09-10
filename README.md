# Smart-Retrieval-System
The Smart Retrieval System leverages AI and vector embeddings to efficiently process and retrieve information from large documents.Additionally, the system integrates a Large Language Model (LLM) to provide contextual understanding, allowing for more accurate and meaningful results by interpreting user queries and document content beyond keyword matching.

## Pre-Requisite
1) Python: Install Python

## Installation
1) Clone the repo, create and activate the environment for the project.
2) Install all required packages from requirements.txt file using command: "pip install -r requirements.txt".

##Built With
1) Python
2) Langchain
3) FAISS vector database
4) Google Generative AI LLM - gemini-1.5-flash
5) Streamlit

## Usage
1) To start the app, run command: "streamlit run smart_retrieval_system.py".
2) The PDF files can be uploaded from "Upload Your PDF" section.
3) The user can chat with the AI assistant from the chat input box provided.

## System Workflow
1) Load the Google API Key using environment variables.
2) Extract text from uploaded PDF documents.
3) Split the text into smaller, manageable chunks.
4) Generate embeddings using Google Generative AI.
5) Store the embeddings in a FAISS vector store.
6) Perform retrieval operations based on user queries.
7) LLM integration to understand and process user queries, enhancing retrieval by providing contextually relevant results beyond keyword matching.


