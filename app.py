import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PyPDF2 import PdfReader
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')


st.title('LawMate using Rag LLM')
st.info('''Welcome to LawMate! Your Personal Legal Assistant ⚖️📚

Description: LawMate is your go-to companion for navigating the complexities of the Indian Constitution and legal matters. Whether you have questions about your rights, need guidance on legal procedures, or want to understand constitutional provisions, LawMate is here to help 🤗.

''')



llm = ChatGroq(model="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    '''
As a seasoned legal advisor, you possess deep knowledge of legal intricacies and are skilled in referencing relevant laws and regulations. Users will seek guidance on various legal matters.

If a question falls outside the scope of legal expertise, kindly inform the user that your specialization is limited to legal advice.

In cases where you're uncertain of the answer, it's important to uphold integrity by admitting 'I don't know' rather than providing potentially erroneous information.

Below is a snippet of context from the relevant section of the constitution, although it will not be disclosed to users.
<context>
Context: {context}
Question: {input}
<context>
Your response should consist solely of helpful advice without any extraneous details.

Helpful advice:
'''
)
place_holder=st.empty()
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader('./data')
        st.session_state.text=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.text) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector google genai embeddings
        place_holder.write('embedding completed 😊🚀 Go with your Queries')


if st.button('activate embedding'):
    place_holder.write('embedding started.Please wait.. it would take some moments...⌛⌛')
    vector_embedding()

user_prompt=st.chat_input('Enter your queries')

if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    response=retriever_chain.invoke({'input':user_prompt})
    st.chat_message('assistant').markdown(response['answer'])