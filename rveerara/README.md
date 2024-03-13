## This README contains information about the code submitted by Rajeev Veeraraghavan for 11-711 assignment 2 in Spring 2024

- The code to create the embedder is present in src/embedder.py
    - The file contains a main method where variables can be set before running the run() function to create the vector store
    - The embedder creates a list of bge-large-en models based that chunk data based on the sizes and overlaps provided
    - The modular functions are invoked in order by the run() function to  create and store embeddings in the path specified in the `vector_store_path` to the run function
    - Similarly, a directory from which to read documents can be specified
    - The reader function will read all .txt, .pdf, .csv and .json files in the directory and chunk them before creating the vector store
- The code to generate answers after creation of the embedder is provided in src/rag.py
    - This file has modular functions to create the QA chain, load the vector store, read questions, generate answers and load the embedding model
    - The functions do_rag() and do_rag_in_chunks() contain code to call all the other functions and write answers to an output file.
    - The vector store path, embedding model, questions file path, output file path can all be specified as arguments to either of the functions do_rag() and do_rag_in_chunks()
- Prior to running either of these files, please install all requirements mentioned in the requirements.txt file preferably in a virtual environment