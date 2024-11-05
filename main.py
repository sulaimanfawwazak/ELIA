from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
import shutil
from dotenv import load_dotenv
import discord
from discord.ext import commands

# Load the .env
load_dotenv()

# Get the bot token
BOT_TOKEN = os.getenv('BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents)

# Path for the Chroma Store
CHROMA_PATH = "chroma"

# Path for the data source
DATA_PATH = "data"

CONFUSED_TEXT = "Sorry, I don't seem to find the information regarding this. Please ask me something else"

# Prompt Template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {query}

If you can not seem to find the answer, please answer with a variation of this text: {confused}
"""

# Function to do embeddings
def get_embedding_function():
  # Instantiate the embedder
  embeddings = OllamaEmbeddings(model="mistral")

  return embeddings

# This part is for populating the Chroma Database
# Function to load the PDF document
def load_documents():
  document_loader = PyPDFDirectoryLoader(DATA_PATH)

  return document_loader.load()

# Function to split the documents into chunks
def split_document(documents: list):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
  )

  return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
  '''
  Function to determine the chunks id
  Format: "Source:Page:Chunk" --> "Document.pdf:14:10"
  '''
  # Format: "data/document.pdf:page:index"

  last_page_id = None
  current_chunk_index = 0

  for chunk in chunks:
    # Get the source
    source = chunk.metadata.get('source')
    # Get the page
    page = chunk.metadata.get('page')

    # Create a page ID with a format of 'Source:Page' --> 'Document.pdf:15'
    # --> 'Document.pdf:15'
    current_page_id = f'{source}:{page}'

    # If the page ID is the same as the last one, increment the index.
    if current_page_id == last_page_id:
      current_chunk_index += 1
    else:
      current_chunk_index = 0

    # Create a chunk ID with a format of 'current_page_id:current_chunk_index' 
    # --> 'Source:Page:Chunk' 
    # --> 'Document.pdf:14:10'
    chunk_id = f"{current_page_id}:{current_chunk_index}"

    # Update last_page_id as current_page_id
    last_page_id = current_page_id

    # Add it to the page meta-data.
    chunk.metadata["id"] = chunk_id

  return chunks

def add_to_chroma(chunks: list[Document]):
  # Load the existing database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

  # Calculate Chunk IDs
  chunks_with_ids = calculate_chunk_ids(chunks)

  # Add or Update the documents.
  existing_items = db.get(include=[])  # IDs are always included by default
  existing_ids = set(existing_items["ids"])
  print(f"Number of existing documents in DB: {len(existing_ids)}")

  # Only add documents that don't exist in the DB.
  new_chunks = []
  for chunk in chunks_with_ids:
    # If the chunk ID is not icluded in the existing ID, append the chunk ID
    if chunk.metadata["id"] not in existing_ids:
      new_chunks.append(chunk)

  if len(new_chunks) > 0:
    print(f"Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
  else:
    print("✅ No new documents to add")

# Function to clear the Chroma store
def clear_chroma():
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Function for the RAG Query
def query_rag(query_text: str):
  # Prepare the Store
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

  # Search for the query in the Chroma Store
  results = db.similarity_search_with_score(query_text, k=5)

  # Arrange the context from the search result
  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

  # Create the prompt from the Prompt Template
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

  prompt = prompt_template.format(context=context_text, query=query_text, confused=CONFUSED_TEXT)
  # print(prompt)

  # Instantiate the model
  model = OllamaLLM(model="mistral")

  # Get the response of the prompt from the model
  response_text = model.invoke(prompt)

  # Get the sources
  sources = [doc.metadata.get("id", None) for doc, _score in results]

  # Format the response 
  # --> Response: 
  # --> ... ... ...
  # --> Sources: 
  # --> ... ... ...
  formatted_response = f"Response: \n{response_text}\nSources: \n{sources}"

  print(formatted_response)

  return response_text

########## BOT THINGS ##########
@client.event
async def on_ready():
    print("The bot is now ready for use!")
    print(".............................")

@client.event
async def on_message(message):
    # This part is to prevent the message loop
    # This checks if the output is from the bot
    # If it is from the bot, then return
    if message.author == client.user:
        return
    
    print(f'[{message.author}]: {message.content}')
    
    # Submit the message as query to the bot
    response_text = query_rag(message.content)

    # Send the response to the server
    await message.channel.send(response_text)

# Function to initialize the RAG, like the main() function of this file
def initialize_rag():
  # Load the document
  documents = load_documents()
  print(f'Loaded documents: {len(documents)}')

  # Separate the documents into chunks
  chunks = split_document(documents)
  print(f'Split documents into {len(chunks)} chunks')

  # Add the chunks into Chroma Store
  add_to_chroma(chunks)
  print(f'Chunks are added to Chroma')

# Call the main function
initialize_rag()

# Run the bot
client.run(BOT_TOKEN)