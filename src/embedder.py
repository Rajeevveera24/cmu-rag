import os, time, argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from chromadb.errors import InvalidDimensionException

BASE_DIR = '/home/raj/nlp/cmu-rag/'
BASE_DIR_TXT_FILES = '/home/raj/nlp/cmu-rag/data/documents/_Combined/'
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
            # print_message("\tLoaded Text file: " + file)
        else:
            print_message("\tSkipped file: " + file + " as it is not a text file")
        if cnt > 0 and (cnt+1) % 5 == 0:
            print_message("Loaded {}/{} files".format(cnt+1, len_documents))
    print_message("Loaded all files from directory: " + dir)
    verbose_flag = False
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=0.2):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap*chunk_size)
    chunks = splitter.split_documents(documents)
    return chunks

def run(doc_path=BASE_DIR_TXT_FILES,
        vector_store_path=VECTOR_DATABASE_PATH + 'llama2',
        embedding_model=OllamaEmbeddings(model='llama2'),
        chunk_size=500,
        chunk_overlap=0.2,
        verbose=False):
    
    global verbose_flag
    verbose_flag = verbose

    documents = read_documents_in_directory(dir=doc_path, verbose=verbose)
    # print(len(documents))
    chunked_documents = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # print(len(chunked_documents))
    create_and_store_embedding(chunked_documents, embedding_model, verbose=verbose, persist_directory=vector_store_path)
    verbose_flag = False

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Embedder')
    parser.add_argument('--docs', type=str, default=BASE_DIR_TXT_FILES, help='Path to the directory containing text files')
    parser.add_argument('--vector', type=str, default=VECTOR_DATABASE_PATH + 'bge', help='Path to the directory where the vector store will be stored')
    parser.add_argument('--embed', type=str, default='bge', help='Embedding model to be used for creating embeddings')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
    parser.add_argument('--chunk_size', type=str, default=500, help='Chunk Size for splitting documents into smaller chunks')
    parser.add_argument('--chunk_overlap', type=str, default=0.2, help='Chunk Overlap for splitting documents into smaller chunks')
    
    args = parser.parse_args()

    return args

def get_bge_embedding_model():
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

if __name__ == "__main__":
    args = parse_arguments()
    documents_dir = args.docs
    vector_store_path = args.vector
    embedding_model_option = args.embed
    verbose = args.verbose
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    vector_store_path = vector_store_path + '/' if vector_store_path[-1] != '/' else vector_store_path
    vector_store_path = vector_store_path + '-' + str(chunk_size) + '-' + str(chunk_overlap)

    embeding_model = get_bge_embedding_model() if embedding_model_option == 'bge' else OllamaEmbeddings(model=embedding_model_option)

    print("Running Embedder with the following options:")
    print("\tDocuments Directory: ", documents_dir)
    print("\tVector Store Path: ", vector_store_path)
    print("\tEmbedding Model: ", embedding_model_option)
    print("\tVerbose: ", verbose)
    print("\tChunk Size: ", chunk_size)
    print("\tChunk Overlap: ", chunk_overlap)

    run(doc_path=documents_dir, vector_store_path=vector_store_path, embedding_model=embeding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap, verbose=verbose)
    
    print("Done")
