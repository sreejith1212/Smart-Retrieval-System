from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain


def read_api():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return google_api_key

def get_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    google_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=google_embedding)
    return vector_store

# New function to optimize the prompt
def optimize_prompt(user_question):
    prompt_template = (
        "You are a helpful assistant. Based on the following context, answer the user's question. "
        "Provide a concise and informative response.\n\n"
        "User's Question: {question}\n"
    )
    optimized_prompt = prompt_template.format(question=user_question)
    return optimized_prompt

def query_ai(vector_store, user_question):
    docs = vector_store.similarity_search(query=user_question, k=5)
    googlellm = ChatGoogleGenerativeAI(
                            model="gemini-1.5-flash",
                            temperature=0,
                            max_tokens=None
                            )
    # Get optimized prompt
    refined_prompt = optimize_prompt(user_question)
    chain = load_qa_chain(llm=googlellm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=refined_prompt)
    return response


if __name__ == "__main__" :

    google_api_key = read_api()

        # set app page layout type
    st.set_page_config(layout="wide")
    

    # create sidebar
    with st.sidebar:        
        page = option_menu(
                            menu_title='Smart App',
                            options=['Assistant'],
                            icons=['person-circle'] ,
                            styles={"container": {"padding": "5!important"},
                                    "icon": {"color": "brown", "font-size": "23px"}, 
                                    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "lightblue"},
                                    "nav-link-selected": {"background-color": "grey"},}  
                        )
        
    col1, col2, col3 = st.columns([2,3,2])
    col2.header(':green[_Smart Retrieval System_] :scroll:')
    

    file_info = st.file_uploader("Upload Your PDF", accept_multiple_files=True)
    if file_info:
        file_extract = get_text(file_info)
        text_chunks = get_text_chunks(file_extract)
        vector_store = get_vector_store(text_chunks)

        user_question = st.chat_input(placeholder="Ask Questions About Your PDF")

        if user_question:
            answer = query_ai(vector_store, user_question)

            st.write(f"User Question: {user_question}")
            # Display the answer
            st.write(f"Answer: {answer}")

  



    


