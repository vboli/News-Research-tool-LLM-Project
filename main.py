import os
import streamlit as st
import pickle
import time
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv





# Load environment variables from .env file (Hugging Face API key)
load_dotenv()

st.title("QueryBot: News Research Tool 📊")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

# Initialize the Hugging Face LLM with the task explicitly set to 'summarization'
llm = HuggingFaceHub(
    repo_id="facebook/bart-large",
    model_kwargs={"temperature": 0.9, "max_length": 500},
    task="summarization"  # Explicitly specify the task
)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save them to the FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_hf = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

# Accept a query from the user
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            try:
                # Use invoke instead of __call__
                result = chain.invoke({"question": query}, return_only_outputs=True)

                # Debugging output
                st.write("Result:", result)
                st.write("Type of result:", type(result))

                # Handle the result being a list
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], dict):
                        # Access the first item in the list and its fields
                        st.header("Answer")
                        st.write(result[0].get("answer", "No answer found"))

                        # Display sources if available
                        sources = result[0].get("sources", "")
                        if sources:
                            st.subheader("Sources:")
                            sources_list = sources.split("\n")  # Split the sources by newline
                            for source in sources_list:
                                st.write(source)
                    else:
                        st.write("Unexpected structure in result list.")

                # If result is a dictionary, handle it
                elif isinstance(result, dict):
                    st.header("Answer")
                    st.write(result.get("answer", "No answer found"))

                    # Display sources if available
                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split("\n")  # Split the sources by newline
                        for source in sources_list:
                            st.write(source)

                else:
                    st.write("Result is in an unrecognized format:", result)

            except Exception as e:
                st.error(f"An error occurred while processing the query: {str(e)}")



