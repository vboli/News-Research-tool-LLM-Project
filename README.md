# QueryBot: News Research Tool

QueryBot is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.


# App Screenshot

![App Screenshot](https://github.com/user-attachments/assets/34d3fc4f-c3c5-4641-b15c-620c7b05206d)


# Architecture flow diagram

![App Screenshot](https://github.com/user-attachments/assets/d45e3b56-7c1d-44d1-a057-910aa5573387)
## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using Huggingface embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's by inputting queries and receiving answers along with source URLs.


## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your HUGGINGFACEHUB_API_TOKEN.
