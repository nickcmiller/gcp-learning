
from google.cloud import aiplatform, storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.vertex import VertexAIEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

import logging

logging.basicConfig(level=logging.INFO)

global_project_id=""
global_region=""

aiplatform.init(project=global_project_id, location=global_region)

def create_bucket(bucket_name: str, region: str=global_region, project_id: str=global_project_id):
    # Create a bucket
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    if not bucket.exists():
        bucket.create(location=region)
        logging.info(f"Bucket {bucket_name} created successfully in region {region}.")
    else:
        logging.info(f"Bucket {bucket_name} already exists.")

    return bucket

def create_index(bucket_name: str, display_name: str, region: str=global_region, project_id: str=global_project_id):

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        gcs_source_uris=[f"gs://{bucket_name}"],
        dimensions=768,  # Adjust dimensions based on your embedding model
        index_update_method="BATCH_UPDATE"
    )

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name
    )
    index.deploy(endpoint=endpoint)
    return {
        "endpoint": endpoint,
        "index": index
    }

# Embed Documents Using LlamaIndex

def embed_documents(data_path: str):
    Settings.embed_model = VertexAIEmbedding(model_name="text-embedding-004")
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index

# Save Embeddings to Vertex AI

def save_embeddings_to_vertex_ai(index: VectorStoreIndex, endpoint: aiplatform.MatchingEngineIndexEndpoint):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", operator=FilterOperator.EQUAL, value="finance")
            ]
        )

        index.save_to_vertex_ai(
            endpoint=endpoint,
            gcs_bucket="your_gcs_bucket",
            filters=filters
        )

# Query the Index

def query_index(index: VectorStoreIndex, query: str):
    query_engine = index.as_query_engine()

    response = query_engine.query("your query string")
    print(response)

if __name__ == "__main__":
    bucket = create_bucket("my_gcs_bucket")

