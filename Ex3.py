# Try to use prompt template similar to this:

import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate  # <--- Imported this

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("PDF Chat Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    st.success("PDF Loaded Successfully")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    # Note: Ensure you have sentence-transformers installed for this model
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )

    # Define the prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Construct the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = chain.invoke(query)
            st.write("### Answer:")
            st.write(answer)