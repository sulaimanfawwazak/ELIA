# ELIA (Edus's Language Interactive Assistant)
## Retrieval-Augmented Generation (RAG) - Based Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using vector embeddings and a document retrieval system. The chatbot can process user queries and provide relevant answers based on pre-embedded documents.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Embedding-based document retrieval for context-aware responses.
- Integration with Ollama's embedding model for efficient query and document embeddings.
- User-friendly chatbot interface for interactive conversation.

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - `discord.py`
  - `langchain`
  - `langchain_community`
  - `langchain_ollama`
  - `langchain_text_splitters`
  - `python-dotenv`

  
You can install the required packages using:

```bash
pip install -r requirements.txt
