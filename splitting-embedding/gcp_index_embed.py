from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

from google.cloud import aiplatform, storage
from google.auth import default
from google.cloud.exceptions import Conflict

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)

def create_bucket(bucket_name, region, project_id):
    credentials, _ = default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    
    try:
        bucket = storage_client.create_bucket(bucket_name, location=region)
        logging.info(f"Bucket {bucket_name} created successfully in region {region}.")
    except Conflict:
        logging.info(f"Bucket {bucket_name} already exists.")
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        bucket = None
    
    return bucket

def query_index(index, query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

class VertexAIIndexManager:
    def __init__(self, project_id, region, index_name, endpoint_name, dimensions=768, approximate_neighbors_count=150):
        self.project_id = project_id
        self.region = region
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.dimensions = dimensions
        self.approximate_neighbors_count = approximate_neighbors_count
        
        aiplatform.init(project=self.project_id, location=self.region)
    
    def create_index(self, bucket_name: str) -> Dict[str, Any]:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            contents_delta_uri=[f"gs://{bucket_name}"],
            dimensions=self.dimensions,
            approximate_neighbors_count=150,
            index_update_method="BATCH_UPDATE"
        )
        
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.endpoint_name,
            enable_private_service_connect=True
        )
        
        deployed_index = endpoint.deploy_index(
            index=index, 
            deployed_index_id=self.index_name
        )
        
        return {
            "endpoint": endpoint,
            "index": index,
            "deployed_index": deployed_index
        }
    
    def embed_documents(self, data_path: str) -> VectorStoreIndex:
        Settings.embed_model = VertexTextEmbedding(model_name="text-embedding-004")
        
        documents = SimpleDirectoryReader(data_path).load_data()
        print(f"Documents: {documents}")
        
        vector_store_index = VectorStoreIndex.from_documents(documents)
        print(f"Vector Store Index: {vector_store_index}")
        
        return vector_store_index
    
    def save_embeddings_to_vertex_ai(self, vector_index: VectorStoreIndex, filter_list: List[MetadataFilter] = None) -> str:
        
        result = vector_index.save_to_vertex_ai(
            endpoint=self.endpoint,
            gcs_bucket=self.bucket_name
        )
        
        return result

if __name__ == "__main__":
    project_id = "split-embeddings-project"
    region = "us-west4"
    bucket_name = "split-embeddings-bucket-189654"
    index_name = "split_embeddings_index"
    endpoint_name = "split_embeddings_endpoint"

    manager = VertexAIIndexManager(
        project_id=project_id,
        region=region,
        index_name=index_name,
        endpoint_name=endpoint_name,
        dimensions=768,
        approximate_neighbors_count=150
    )
    
    if False:
        bucket = create_bucket(bucket_name, region, project_id)
        print(f"Bucket: {bucket}")
    
    if False:
        index = manager.create_index(bucket_name)
        print(f"Index Data: {index}")

    if True:
        import os
        cwd = os.getcwd()
        pdf_file_path = f"{cwd}/file_directory"
        print(f"PDF File Path: {pdf_file_path}")
        vector_store_index = manager.embed_documents(pdf_file_path)
        print(f"Vector Store Index: {vector_store_index}")
    
    if False:
        manager.save_embeddings_to_vertex_ai(vector_store_index, bucket_name)
    
    if False:
        # Example usage of query_index
        index = index_data["index"]
        query = "What is this about?"
        response = query_index(index, query)
        print(f"Query Response: {response}")

