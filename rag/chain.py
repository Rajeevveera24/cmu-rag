import os, time
import chromadb

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain import hub

from chromadb.errors import InvalidDimensionException

DATABASE_PATH = '/home/raj/nlp/cmu-rag/rag/chroma/txt/'
MODEL_NAMES = ['tinyllama', 'llama2', 'gemma', 'mistral', 'neural-chat', 'openchat']
VECTOR_STORE_DIRECTORIES = [DATABASE_PATH + embedding_name for embedding_name in MODEL_NAMES]
ANNOTATION_DIR = '/home/raj/nlp/cmu-rag/annotation/test/history/'
ANNOTATION_FILE = ANNOTATION_DIR + 'questions.txt'
PROMPT_MESSAGE_LLAMA2 = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use as few words as possible and keep the answer concise. Do not mention the context in your response.
    Question: {question} 
    Context: {context} 
    Answer:"""




def load_vector_store(dir, embedding_model = OllamaEmbeddings()):
    try:
        vector_store = Chroma(persist_directory=dir, embedding_function=embedding_model)
    except InvalidDimensionException:
        vector_store = Chroma(persist_directory=dir, embedding_function=embedding_model, force=True)
    return vector_store


def create_chain(vector_store, model_name = 'llama2', prompt_message = PROMPT_MESSAGE_LLAMA2):
    
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    rag_prompt_llama.messages[0].prompt.template = prompt_message
    llm = Ollama(model = model_name)    

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vector_store.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_llama
        | llm
        | StrOutputParser()
    )

    return qa_chain


def get_questions(file_name = ANNOTATION_FILE):
    
    if not file_name.endswith('questions.txt'):
        raise ValueError("Invalid file name")
    
    questions = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            questions.append(line.strip())
    
    return questions


def generate_answers(qa_chain, questions=get_questions()):
    if not questions:
        raise ValueError("No questions to answer")
    if not qa_chain:
        raise ValueError("No qa_chain to answer questions")
    
    answers = []
    for question in questions:
        if not question:
            continue
        answer = dict()
        answer_raw = qa_chain.invoke(question)
        answer["raw"] = answer_raw
        num_lines = answer_raw.count('\n')
        answer["num_lines"] = num_lines
        lines = answer_raw.split('\n')
        if num_lines == 0:
            answer["processed"] = lines[0]
        else:
            answer_lines = []
            for line in lines:
                if "i don't know" not in line.lower():
                    answer_lines.append(line)
            answer["processed"] = " ".join(answer_lines)
        answers.append(answer)

    return answers


def write_answers(answers, file_name, append = False):
    try:
        with open(file_name, 'a' if append else 'w') as f:
            for answer in answers:
                f.write(answer["processed"] + '\n')
    except Exception as e:
        raise Exception("Error writing answers to file: " + str(e))

def get_hugging_face_embedding_model():
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )
    return hf

def do_rag(vector_store_path=DATABASE_PATH+'bge-large-en',
        embedding_model=get_hugging_face_embedding_model(),
        model_name='llama2',
        questions_file_name=ANNOTATION_DIR+'questions.txt',
        answers_file_name=ANNOTATION_DIR+'answers.txt',
        append=False):
    vector_store = load_vector_store(vector_store_path, embedding_model)
    qa_chain = create_chain(vector_store, model_name)
    questions = get_questions(file_name=questions_file_name)
    answers = generate_answers(qa_chain, questions)
    write_answers(answers, answers_file_name, append=append)

if __name__ == "__main__":
    do_rag()
    print("Done")
