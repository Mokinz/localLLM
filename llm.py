"""IMPORTS"""

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index import StorageContext, ServiceContext
import os

def list_files(directory):
    file_paths = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

def setup_llm_model(model_name, temperature=0.1, context_window=3900):
    """CHOOSING MODEL"""
    if model_name == "Llama2-13B":
        model_path = ".\\llms\\llama-2-13b-chat\\llama-2-13b-chat.Q5_K_M.gguf"
    elif model_name == "Llama2-7B":
        model_path = ".\\llms\\llama-2-7b-chat\\llama-2-7b-chat.Q5_K_M.gguf"
    elif model_name == "Llama3-8B":
        model_path = ".\\llms\\llama-3-8b\\Meta-Llama-3-8B-Instruct-Q6_K.gguf"

    """LAMACPP SETUP"""
    llm = LlamaCPP(

    model_path= model_path,
    temperature=temperature,
    max_new_tokens=256,
    context_window=context_window,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)
    return llm


def generate_vector_space(context: str):
    if context == "CRM":
        docs_path = ".\\instance\\crm"
    elif context == "ERP":
        docs_path = ".\\instance\\erp"
    elif context == "Medical":
        docs_path = ".\\instance\\medical"
    elif context == "TERG":
        docs_path = ".\\instance\\TERG"

    context_window = 2048
    num_output = 256

    llm = setup_llm_model(model_name="Llama3-8B")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


    files_list = list_files(docs_path)
    files = list(filter(lambda path: os.path.splitext(path)[1].lower() != '.aspx', files_list))

    documents = SimpleDirectoryReader(input_files=files).load_data()

    db = chromadb.PersistentClient(path=".\\chroma_db")
    chroma_collection = db.get_or_create_collection(context)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        context_window=context_window,
        num_output=num_output,
        chunk_size=256, 
        chunk_overlap=0.15
    )
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )


def generate_llama_response(prompt: str, context: str, model_name: str):
    context_window = 2048
    num_output = 256

    llm = setup_llm_model(model_name=model_name)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        context_window=context_window,
        num_output=num_output,
        chunk_size=256, 
        chunk_overlap=0.15
    )

    db2 = chromadb.PersistentClient(path=".\\chroma_db")
    chroma_collection = db2.get_or_create_collection(context)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, service_context=service_context
    )

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    response_iter = chat_engine.stream_chat(prompt)

    return response_iter.response_gen
