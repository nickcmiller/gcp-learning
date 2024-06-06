from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

from google.cloud import aiplatform, storage
from google.auth import default
from google.cloud.exceptions import Conflict

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)

def get_project_id_by_name(target_project_name):
       """Retrieve the project ID given a project name."""
       client = resource_manager.Client()
       for project in client.list_projects():
           if project.name == target_project_name:
               return project.project_id
       return None

def create_bucket(index_bucket_name, region, project_id):
    credentials, _ = default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    
    try:
        bucket = storage_client.create_bucket(index_bucket_name, location=region)
        logging.info(f"Bucket {index_bucket_name} created successfully in region {region}.")
    except Conflict:
        logging.info(f"Bucket {index_bucket_name} already exists.")
        bucket = storage_client.bucket(index_bucket_name)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        bucket = None
    
    return bucket

def query_index(index, query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

class VertexAIIndexManager:
    def __init__(
        self, 
        project_id, 
        region, 
        index_id, 
        endpoint_id, 
        endpoint_name, 
        index_bucket_name, 
        staging_bucket_name, 
        embed_model_name, 
        dimensions=768, 
        approximate_neighbors_count=150
    ):
        self.project_id = project_id
        self.region = region
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.endpoint_name = endpoint_name
        self.index_bucket_name = index_bucket_name
        self.staging_bucket_name = staging_bucket_name
        self.embed_model_name = embed_model_name
        self.dimensions = dimensions
        self.approximate_neighbors_count = approximate_neighbors_count
        
        aiplatform.init(
            project=self.project_id, 
            location=self.region, 
            staging_bucket=f"gs://{self.staging_bucket_name}"
        )
    
    def create_index(self):
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_id,
            contents_delta_uri=[f"gs://{self.index_bucket_name}"],
            dimensions=self.dimensions,
            approximate_neighbors_count=self.approximate_neighbors_count,
            index_update_method="BATCH_UPDATE"
        )
        return index
    
    def deploy_index_and_endpoint():
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.endpoint_name,
            enable_public_endpoint=True
        )
        
        deployed_index = endpoint.deploy_index(
            index=index, 
            deployed_index_id=self.index_id
        )

        return {
            "endpoint": endpoint,
            "deployed_index": deployed_index
        }
    
    def initialize_vector_store(self, documents, endpoint_id):
        print(f"project_id: {self.project_id}\nregion: {self.region}\nindex_id: {self.index_id}\nendpoint_id: {endpoint_id}")
        print(f"{type(self.project_id)}, {type(self.region)}, {type(self.index_id)}, {type(endpoint_id)}")
        
        vector_store = VertexAIVectorStore(
            project_id=self.project_id,
            region=self.region,
            index_id=self.index_id,
            endpoint_id=endpoint_id,
            gcs_bucket_name=self.staging_bucket_name
        )

        # configure embedding model
        embed_model = VertexTextEmbedding(
            model_name=self.embed_model_name,
            project=self.project_id,
            location=self.region,
        )

        # setup the index/query process, ie the embedding model (and completion if used)
        Settings.embed_model = embed_model

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        vector_store_index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )

        return vector_store_index


if __name__ == "__main__":
    project_id = "545919198487"
    region = "us-west4"
    index_id = "491314571848450048"
    index_bucket_name = "split-embeddings-bucket-189654"
    staging_bucket_name = "split-embeddings-staging-bucket-189654"
    endpoint_name = "split_embeddings_endpoint"
    endpoint_id = "822469881948536832"
    embed_model_name = "textembedding-gecko@003"

    manager = VertexAIIndexManager(
        project_id=project_id,
        region=region,
        index_id=index_id,
        index_bucket_name=index_bucket_name,
        staging_bucket_name=staging_bucket_name,
        endpoint_name=endpoint_name,
        endpoint_id=endpoint_id,
        embed_model_name=embed_model_name,
        dimensions=768,
        approximate_neighbors_count=150
    )
    
    if False:
        index_bucket = create_bucket(index_bucket_name, region, project_id)
        print(f"Index Bucket: {index_bucket}")

    if True:
        staging_bucket = create_bucket(staging_bucket_name, region, project_id)
        print(f"Staging Bucket: {staging_bucket}")

    
    
    if True:
        index = manager.create_index()
        print(f"Index Data: {index}")

    if True:
        index_endpoint = manager.deploy_index_and_endpoint()
        print(f"Index Endpoint: {index_endpoint}")


    if False:
        project_id = "545919198487"
        region = "us-west4"
        index_endpoint_id = "822469881948536832"

        client = aiplatform.gapic.IndexEndpointServiceClient(client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})
        index_endpoint = client.index_endpoint_path(project=project_id, location=region, index_endpoint=index_endpoint_id)

        print(f"Index Endpoint: {index_endpoint}")

    if True: 
        import os
        cwd = os.getcwd()
        pdf_file_path = f"{cwd}/file_directory/"
        documents = SimpleDirectoryReader(pdf_file_path).load_data()   
        vector_store_index = manager.initialize_vector_store(documents, endpoint_id)
        print(f"Vector Store Index: {vector_store_index}")
    
    if False:
        # Example usage of query_index
        index = index_data["index"]
        query = "What is this about?"
        response = query_index(index, query)
        print(f"Query Response: {response}")

