import os, time
import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from chromadb.errors import InvalidDimensionException

BASE_DIR = '/home/raj/nlp/cmu-rag/'
BASE_DIR_TXT_FILES = '/home/raj/nlp/cmu-rag/data/documents/_Combined/'
VECTOR_DATABASE_PATH = '/home/raj/nlp/cmu-rag/chroma_vector_database/'
EMBEDDING_OPTIONS = ['llama2', 'everythinglm', 'mistral', 'neural-chat', 'openchat', 'BGE']

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 0.2

verbose_flag = False

def print_message(message : str) -> None:
    global verbose_flag
    if verbose_flag:
        print(message)

def create_and_store_embedding(documents, embedding_model, persist_directory=VECTOR_DATABASE_PATH, verbose = False):
    
    if not documents or not embedding_model:
        raise Exception("Documents and embedding model are required for creating embeddings")
    
    global verbose_flag
    verbose_flag = verbose

    start_time = time.time()
    print_message("Creating embeddings for: {}".format(embedding_model.__class__))
    
    vector_store = None
    
    try:
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=persist_directory)
        vector_store.persist()
        print_message("Embeddings created and stored for: {}".format(embedding_model.__class__))
    except InvalidDimensionException as e:
        print("Invalid dimension for embedding: ", embedding_model.__class__)
    except Exception as e:
        print("Error: ", e)
    finally:
        end_time = time.time()
    
    print_message("Operation completed in: " + str(end_time - start_time) + " seconds")
    verbose_flag = False

def get_documents_from_txt_file(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

def get_documents_from_json_file(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[]',
        text_content=False)
    docs = loader.load()
    return docs

def read_documents_in_directory(dir=BASE_DIR_TXT_FILES, verbose=False):
    global verbose_flag
    verbose_flag = verbose

    print_message("Loading all files from directory: " + dir + "...")
    documents = []
    len_documents = len(os.listdir(dir))
    for cnt, file in enumerate(sorted(os.listdir(dir))):
        if file.endswith(".txt"):
            documents.extend(get_documents_from_txt_file(dir + file))
            # print_message("\tLoaded Text file: " + file)
        if file.endswith(".pdf"):
            # print_message("\tSkipping PDF file: " + file)
            continue
        if file.endswith(".json"):
            documents.extend(get_documents_from_json_file(dir + file))
            # print_message("\tLoaded JSON file: " + file)
        if cnt > 0 and (cnt+1) % 10 == 0:
            print_message("Loaded {}/{} files".format(cnt+1, len_documents))
    print_message("Loaded all files from directory: " + dir)
    verbose_flag = False
    return documents

def chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    print_message("Chunking documents into size "+ str(chunk_size) + " with overlap of " + str(chunk_overlap*chunk_size))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap*chunk_size)
    chunks = splitter.split_documents(documents)
    return chunks

def run(doc_path=BASE_DIR_TXT_FILES,
        vector_store_path=VECTOR_DATABASE_PATH + 'llama2',
        embedding_model=OllamaEmbeddings(model='llama2'),
        verbose=False,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    start_time = time.time()
    global verbose_flag
    verbose_flag = verbose

    documents = read_documents_in_directory(dir=doc_path, verbose=verbose)
    # print(len(documents))
    chunked_documents = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # print(len(chunked_documents))
    create_and_store_embedding(chunked_documents, embedding_model, verbose=verbose, persist_directory=vector_store_path)
    verbose_flag = False
    end_time = time.time()
    print("Operation completed in: " + str(end_time - start_time) + " seconds")

if __name__ == "__main__":

    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    
    hf_bge_embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

    chunk_sizes = [250, 500, 750, 1000, 1500, 2000]
    chunk_overlaps = [0.1, 0.2, 0.3, 0.4]
    
    for chunk_size in chunk_sizes:
       for chunk_overlap in chunk_overlaps:
           run(vector_store_path=VECTOR_DATABASE_PATH + 'bge-text-enhanced-' + str(chunk_size)+'-'+str(chunk_overlap), embedding_model=hf_bge_embedding_model, verbose=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            run(vector_store_path=VECTOR_DATABASE_PATH + 'llama2-text-enhanced-' + str(chunk_size)+'-'+str(chunk_overlap), embedding_model=OllamaEmbeddings(model = 'llama2'), verbose=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("Done")
    