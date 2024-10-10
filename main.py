# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
import shutil

from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv('./.env')

BOT_TOKEN = os.getenv('BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents)

CHROMA_PATH = "chroma"
DATA_PATH = "data"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to do embeddings
def get_embedding_function():
  embeddings = OllamaEmbeddings(model="mistral")
  return embeddings

# This part is for populating the Chroma Database
def load_documents():
  document_loader = PyPDFDirectoryLoader(DATA_PATH)

  return document_loader.load()

def split_document(documents: list):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
  )

  return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
  # Format: "data/document.pdf:page:index"

  last_page_id = None
  current_chunk_index = 0

  for chunk in chunks:
    source = chunk.metadata.get("source")
    page = chunk.metadata.get("page")
    current_page_id = f"{source}:{page}"

    # If the page ID is the same as the last one, increment the index.
    if current_page_id == last_page_id:
      current_chunk_index += 1
    else:
      current_chunk_index = 0

    # Calculate the chunk ID.
    chunk_id = f"{current_page_id}:{current_chunk_index}"
    last_page_id = current_page_id

    # Add it to the page meta-data.
    chunk.metadata["id"] = chunk_id

  return chunks

def add_to_chroma(chunks: list[Document]):
  # Load the existing database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

  # Calculate Page IDs
  chunks_with_ids = calculate_chunk_ids(chunks)

  # Add or Update the documents.
  existing_items = db.get(include=[])  # IDs are always included by default
  existing_ids = set(existing_items["ids"])
  print(f"Number of existing documents in DB: {len(existing_ids)}")

  # Get existing IDs
  # existing_items = db._collection.get(include=["ids"])
  # existing_ids = set(existing_items["ids"])
  # print(f"Number of existing documents in DB: {len(existing_ids)}")

  # Only add documents that don't exist in the DB.
  new_chunks = []
  for chunk in chunks_with_ids:
    if chunk.metadata["id"] not in existing_ids:
      new_chunks.append(chunk)

  if len(new_chunks):
    print(f"👉 Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
  else:
    print("✅ No new documents to add")

def clear_store():
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# RAG Query
def query_rag(query_text: str):
  # Prepare the Store
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

  # Search the DB.
  results = db.similarity_search_with_score(query_text, k=5)

  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  # print(prompt)

  model = OllamaLLM(model="mistral")
  response_text = model.invoke(prompt)

  sources = [doc.metadata.get("id", None) for doc, _score in results]
  formatted_response = f"Response: \n{response_text}\nSources: \n{sources}"
  print(formatted_response)
  return response_text


@client.event
async def on_ready():
    print("The bot is now ready for use!")
    print(".............................")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    print(f'[{message.author}] {message.content}')
    
    response_text = query_rag(message.content)
    await message.channel.send(response_text)

def initialize_rag():
  documents = load_documents()
  print(f'Loaded documents: {len(documents)}')

  chunks = split_document(documents)
  print(f'Split documents into {len(chunks)} chunks')

  add_to_chroma(chunks)
  print(f'Chunks are added to Chroma')

initialize_rag()

client.run(BOT_TOKEN)