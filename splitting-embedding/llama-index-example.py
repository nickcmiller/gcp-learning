from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = Groq(model="llama3-70b-8192")

documents = SimpleDirectoryReader(
        input_files=["./investing_in_unknown_and_unknowable.pdf"]
    ).load_data()

vector_index = VectorStoreIndex.from_documents(documents)

def query_pdf(query, index):

    query_engine = index.as_query_engine()

    response = query_engine.query(query)
    print(f"Response: {response}\n\n")

    return response

if __name__ == "__main__":
    query_pdf("What is this about?", vector_index)
    query_pdf("What is the author?", vector_index)
    query_pdf("What are the main themes of this document?", vector_index)


