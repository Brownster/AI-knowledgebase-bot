#!pip install openai -q
#!pip install langchain -q
#!pip install chromadb -q
#!pip install tiktoken -q
#!pip install pypdf -q
#!pip install unstructured[local-inference] -q
#!pip install gradio -q

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader

import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name="gpt-4")

# Check if any document loaders are specified
loaders = []
if os.path.exists('./docs/'):
    pdf_loader = DirectoryLoader('./docs/', glob="**/*.pdf")
    excel_loader = DirectoryLoader('./docs/', glob="**/*.txt")
    word_loader = DirectoryLoader('./docs/', glob="**/*.docx")
    loaders = [pdf_loader, excel_loader, word_loader]

# Load documents and embed them if loaders are specified
documents = []
if loaders:
    # Load documents from loaders
    for loader in loaders:
        documents.extend(loader.load())
    # Split documents into chunks and embed them
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    persist_directory = './db'
    embeddings = OpenAIEmbeddings()
    if documents:
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
        vectorstore.add_documents(documents)
        vectorstore.persist()
# Initialise Langchain - Conversation Retrieval Chain
if loaders:
    # Use the document-based vectorstore if documents were loaded
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
else:
    # Use an empty vectorstore if no documents were loaded
    persist_directory = './db'
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())

# Front end web app
import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    
    def user(user_message, history):
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
