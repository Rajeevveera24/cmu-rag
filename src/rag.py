

import argparse

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.document_compressors import LLMChainFilter, EmbeddingsFilter, FlashrankRerank, DocumentCompressorPipeline, LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

from langchain import hub

from chromadb.errors import InvalidDimensionException

VECTOR_DATABASES_DIR_PATH = '/home/raj/nlp/cmu-rag/chroma_vector_database/'
VECTOR_STORE_DEFAULT = 'bge-500-0.2'
EMMBEDDING_DEFAULT = 'bge'
ANNOTATION_DIR = '/home/raj/nlp/cmu-rag/rveerara/system_outputs/'
QUESTIONS_FILE = ANNOTATION_DIR + 'questions.txt'
ANSWERS_FILE = ANNOTATION_DIR + 'answers.txt'
PROMPT_MESSAGE_LLAMA2 = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Use as few words as possible and keep the answer concise. Do not mention the context in your response.
    Question: {question} 
    Context: {context} 
    Answer:"""

def load_vector_store(dir, embedding_model = OllamaEmbeddings()):
    try:
        vector_store = Chroma(persist_directory=dir, embedding_function=embedding_model)
    except InvalidDimensionException:
        vector_store = Chroma(persist_directory=dir, embedding_function=embedding_model, force=True)
    return vector_store


def create_chain(vector_store, inference_model = Ollama(model='llama2'), prompt_message = PROMPT_MESSAGE_LLAMA2, embedding_model = OllamaEmbeddings()):
    
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    rag_prompt_llama.messages[0].prompt.template = prompt_message
    llm = inference_model

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.65)

    llm_compressor = LLMChainExtractor.from_llm(llm)

    flashrankRerank = FlashrankRerank(top_n = 7)

    pipeline_compressor = DocumentCompressorPipeline(transformers=[embeddings_filter, llm_compressor])

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    qa_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_llama
        | llm
        | StrOutputParser()
    )

    return qa_chain


def get_questions(file_name = QUESTIONS_FILE):
    
    if not file_name.endswith('questions.txt'):
        raise ValueError("Invalid file name")
    
    questions = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            questions.append(line.strip())
    
    return questions


def generate_answers(qa_chain, questions):
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

def do_rag_in_chunks(vector_store_path=VECTOR_DATABASES_DIR_PATH+VECTOR_STORE_DEFAULT,
        embedding_model = None,
        inference_model=Ollama(model='llama2'),
        questions_file_name=QUESTIONS_FILE,
        answers_file_name=ANSWERS_FILE,
        append=False,
        questions_to_process_at_once=200,):
    if not embedding_model:
        raise ValueError("Invalid embedding model")
    vector_store = load_vector_store(vector_store_path, embedding_model)
    qa_chain = create_chain(vector_store, inference_model=inference_model, embedding_model=embedding_model)
    questions = get_questions(file_name=questions_file_name)
    num_questions = len(questions)
    for i in range(0, num_questions, questions_to_process_at_once):
        questions_chunk = questions[i:min(i+questions_to_process_at_once, num_questions)]
        answers = generate_answers(qa_chain, questions_chunk)
        write_answers(answers, answers_file_name, append=True) if i > 0 else write_answers(answers, answers_file_name, append=append)
        print(f"Processed {i+questions_to_process_at_once} questions out of {num_questions}")


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

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='RAG Chain')
    parser.add_argument('--vector', type=str, default=VECTOR_DATABASES_DIR_PATH+VECTOR_STORE_DEFAULT, help='Path to the directory containing the vector store')
    parser.add_argument('--embed', type=str, default=EMMBEDDING_DEFAULT, help='Embedding model to be used for loading embeddings')
    parser.add_argument('--model', type=str, default='llama2', help='Model name to be used for read documents and generate answers')
    parser.add_argument('--questions', type=str, default=QUESTIONS_FILE, help='Path to the file containing questions')
    parser.add_argument('--answers', type=str, default=ANSWERS_FILE, help='Path to the file where answers will be written')
    parser.add_argument('--append', type=bool, default=False, help='Append answers to the file')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    print("Starting RAG Chain")

    args = parse_arguments()
    vector_store_path = args.vector
    embedding_model_option = args.embed
    model_name = args.model
    questions_file_name = args.questions
    answers_file_name = args.answers
    append = args.append

    embedding_model = get_hugging_face_embedding_model() if embedding_model_option == 'bge' else OllamaEmbeddings(model=embedding_model_option)
    inference_model = Ollama(model=model_name)
    
    print("Starting RAG Chain with the following parameters:")
    print(f"\tVector Store Path: {vector_store_path}")
    print(f"\tEmbedding Model: {embedding_model.__class__.__name__}")
    print((f"\tInference Model: {inference_model.__class__.__name__}") + (f" ({model_name})" if model_name else ""))
    print(f"\tQuestions File: {questions_file_name}")
    print(f"\tAnswers File: {answers_file_name}")
    print(f"\tAppend: {append}")

    do_rag_in_chunks(vector_store_path=vector_store_path,
        embedding_model=embedding_model,
        inference_model=inference_model,
        questions_file_name=questions_file_name,
        answers_file_name=answers_file_name,
        append=append)
    
    print("Done")
