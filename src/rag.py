import os, time, argparse
import chromadb

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain import hub

from chromadb.errors import InvalidDimensionException

DATABASE_PATH = '/home/raj/nlp/cmu-rag/chroma_vector_database/'
MODEL_NAMES = ['llama2', 'mistral', 'neural-chat', 'openchat']
VECTOR_STORE_DIRECTORIES = [DATABASE_PATH + embedding_name for embedding_name in MODEL_NAMES] # Not relevant anymore
ANNOTATION_DIR = '/home/raj/nlp/cmu-rag/rveerara/system_outputs/'
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

def do_rag_in_chunks(vector_store_path=DATABASE_PATH+'bge-large-en-text-only',
        embedding_model=get_hugging_face_embedding_model(),
        model_name='llama2',
        questions_file_name=ANNOTATION_DIR+'questions.txt',
        answers_file_name=ANNOTATION_DIR+'answers.txt',
        append=False,
        question_chunk_size=50,):
    vector_store = load_vector_store(vector_store_path, embedding_model)
    qa_chain = create_chain(vector_store, model_name)
    questions = get_questions(file_name=questions_file_name)
    num_questions = len(questions)
    for i in range(0, num_questions, question_chunk_size):
        questions_chunk = questions[i:min(i+question_chunk_size, num_questions)]
        answers = generate_answers(qa_chain, questions_chunk)
        write_answers(answers, answers_file_name, append=True) if i > 0 else write_answers(answers, answers_file_name, append=append)
        print(f"Processed {i+question_chunk_size} questions out of {num_questions}")

# def parse_args():
#     parser = argparse.ArgumentParser(description="Get read and write file names")

#     parser.add_argument('--dir', metavar='N', type=int, nargs='+',
#                         help='an integer for the accumulator')
#     parser.add_argument('--qf', metavar='N', type=int, nargs='+',
#                         help='an integer for the accumulator')

#     parser.add_argument('--af', dest='accumulate', action='store_const',
#                         const=sum, default=max,
#                         help='sum the integers (default: find the max)')
    
#     args = parser.parse_args()
#     return args.dir, args.qf, args.af


if __name__ == "__main__":
    # annotation_dir = '/home/raj/nlp/cmu-rag/rveerara/system_outputs/'
    annotation_dir = '/home/raj/nlp/cmu-rag/rveerara/system_outputs/'
    q_file = annotation_dir + 'questions.txt'
    a_file = annotation_dir + 'system_output_3.txt'
    # q_file, a_file = None, None
    # try:
    #     annotation_dir, q_file, a_file = parse_args()
    # except Exception as e:
    #     print("Error parsing arguments: ", e)
    #     print("Using default file names")
    #     exit(1)
    # do_rag_in_chunks(vector_store_path=DATABASE_PATH+'llama2-text-only',
    #     embedding_model=OllamaEmbeddings(model='llama2'),
    #     model_name='llama2',
    #     questions_file_name=q_file,
    #     answers_file_name=a_file)
    
    # embedding_chunk_sizes = [250, 500, 750, 1000, 1500, 2000]
    # embeding_chunk_overlaps = [0.1, 0.2, 0.3, 0.4]
    embedding_chunk_sizes = [1000]
    embeding_chunk_overlaps = [0.2]
    embedding = 'bge-all'
    do_rag_in_chunks(
                    vector_store_path='/home/raj/nlp/cmu-rag/chroma_vector_database/bge-text-enhanced-2000-0.4',
                    embedding_model=get_hugging_face_embedding_model(),
                    model_name='openchat',
                    questions_file_name=q_file,
                    answers_file_name=a_file,)
    # for chunk_size in embedding_chunk_sizes:
    #     for chunk_overlap in embeding_chunk_overlaps:
    #         for model in MODEL_NAMES:
    #             print("Processing model: ", model, "with chunk size: ", chunk_size, "and overlap: ", chunk_overlap)
    #             vector_store_path = DATABASE_PATH+embedding+'-'+str(chunk_size)+'-'+str(chunk_overlap)
    #             do_rag_in_chunks(
    #                 vector_store_path=vector_store_path,
    #                 embedding_model=get_hugging_face_embedding_model(),
    #                 model_name=model,
    #                 questions_file_name=q_file,
    #                 answers_file_name=annotation_dir+model+'-BGE-all-' + str(chunk_size) + '-' + str(chunk_overlap) + '.txt',)
    #             # do_rag_in_chunks(
    #             #     vector_store_path=DATABASE_PATH+'llama2-text-only',
    #             #     embedding_model=OllamaEmbeddings(model='llama2'),
    #             #     model_name=model,
    #             #     questions_file_name=q_file,
    #             #     answers_file_name=annotation_dir+model+'-LLAMA2-text-only-answers.txt',)
    print("Done")
