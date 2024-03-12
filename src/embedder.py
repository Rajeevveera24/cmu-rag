import os, time
import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from chromadb.errors import InvalidDimensionException

BASE_DIR = '/home/raj/nlp/cmu-rag/'
BASE_DIR_TXT_FILES = '/home/raj/nlp/cmu-rag/data/documents/combined_txt_files/'
VECTOR_DATABASE_PATH = '/home/raj/nlp/cmu-rag/chroma_vector_database/'
EMBEDDING_OPTIONS = ['llama2', 'everythinglm', 'mistral', 'neural-chat', 'openchat', 'BGE']

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

def read_documents_in_directory(dir=BASE_DIR_TXT_FILES, verbose=False):
    global verbose_flag
    verbose_flag = verbose

    print_message("Loading all files from directory: " + dir + "...")
    documents = []
    len_documents = len(os.listdir(dir))
    for cnt, file in enumerate(os.listdir(dir)):
        if file.endswith(".txt"):
            documents.extend(get_documents_from_txt_file(dir + file))
            print_message("\tLoaded Text file: " + file)
        if cnt > 0 and cnt % 5 == 0:
            print_message("Loaded {}/{} files".format(cnt, len_documents))
    print_message("Loaded all files from directory: " + dir)
    verbose_flag = False
    return documents

def chunk_documents(documents, chunk_size=300, chunk_overlap=0.2):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap*chunk_size)
    chunks = splitter.split_documents(documents)
    return chunks

def run(doc_path=BASE_DIR_TXT_FILES,
        vector_store_path=VECTOR_DATABASE_PATH + 'llama2',
        embedding_model=OllamaEmbeddings(model='llama2'),
        verbose=False):
    
    global verbose_flag
    verbose_flag = verbose

    documents = read_documents_in_directory(dir=doc_path, verbose=verbose)
    # print(len(documents))
    chunked_documents = chunk_documents(documents)
    # print(len(chunked_documents))
    create_and_store_embedding(chunked_documents, embedding_model, verbose=verbose, persist_directory=vector_store_path)
    verbose_flag = False

if __name__ == "__main__":

    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_bge_embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

    # run(verbose=True)
    
    run(vector_store_path=VECTOR_DATABASE_PATH + 'bge-large-en-text-only', embedding_model=hf_bge_embedding_model, verbose=True)
    run(vector_store_path=VECTOR_DATABASE_PATH + 'llama2-text-only', embedding_model=OllamaEmbeddings('llama2'), verbose=True)
    
    print("Done")
    