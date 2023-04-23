Langchain Document-based Conversational AI

This is a document-based conversational AI system powered by GPT-4 and Langchain. It can answer questions based on the contents of documents stored in the docs directory. The application uses Gradio to provide a user interface for interacting with the conversational AI.
Features

    Load documents in PDF, TXT, and DOCX formats
    Index documents using OpenAI embeddings
    Store document embeddings in Chroma for efficient search
    Retrieve information from the documents using a GPT-4 model
    Provide a user-friendly Gradio frontend for interacting with the AI

Installation

To use this script, you will need to install several Python libraries. You can do this by running the following command:

bash

pip install openai langchain chromadb tiktoken pypdf unstructured[local-inference] gradio

Usage

    Store your documents in a folder named docs in the same directory as the script.
    Set your OpenAI API key in the script:

python

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    Run the script:

bash

python langchain_conversational_ai.py

    Open the Gradio web app in your browser and start interacting with the conversational AI.

Code Overview

The main components of this script include:

    Importing necessary libraries
    Initializing the GPT-4 model
    Defining and loading document loaders for PDF, TXT, and DOCX formats
    Loading and embedding documents with OpenAI embeddings
    Initializing the Langchain conversational retrieval chain
    Defining the Gradio user interface and related functions
    Launching the Gradio web app
