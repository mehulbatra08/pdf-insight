import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, "Answer is not available in the context".\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def display_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("**Reply:**", response["output_text"])
    except Exception as e:
        st.error(f"Error processing the question: {e}")

# Main function
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide", initial_sidebar_state="expanded")
    st.title("Chat with PDF using Gemini 💁")
    st.write("Upload your PDF files, process them, and ask questions to get detailed answers.")

    st.sidebar.title("Menu")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.sidebar.success("PDF processing completed successfully!")
                else:
                    st.sidebar.error("No text found in the PDF files.")
        else:
            st.sidebar.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        with st.spinner("Searching for answers..."):
            user_input(user_question)

if __name__ == "__main__":
    main()
