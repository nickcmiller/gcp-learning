from google.cloud import aiplatform

from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

import logging
from dotenv import load_dotenv
import os
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BUCKET_URI = os.getenv("GCS_BUCKET_URI")
VS_DIMENSIONS = os.getenv("VS_DIMENSIONS")
APPROXIMATE_NEIGHBORS_COUNT = os.getenv("APPROXIMATE_NEIGHBORS_COUNT")
VS_INDEX_NAME = os.getenv("VS_INDEX_NAME")
VS_INDEX_ENDPOINT_NAME = os.getenv("VS_INDEX_ENDPOINT_NAME")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")

aiplatform.init(project=PROJECT_ID, location=REGION)

def create_index(
    index_name:str, 
    dimensions:int, 
    distance_measure_type:str, 
    shard_size:str, 
    index_update_method:str, 
    approximate_neighbors_count:int
) -> aiplatform.MatchingEngineIndex:
    """
        Creates a Vector Search index.

        Args:
        index_name (str): The name of the index to be created.
        dimensions (int): The number of dimensions for the index.
        distance_measure_type (str): The type of distance measure to use.
        shard_size (str): The size of the shard.
        index_update_method (str): The method to use for updating the index.
        approximate_neighbors_count (int): The approximate number of neighbors to consider.

        Returns:
        aiplatform.MatchingEngineIndex: The created Vector Search index.
    """
    # Check if the index exists
    index_names = [
        index.resource_name
        for index in aiplatform.MatchingEngineIndex.list(
            filter=f"display_name={index_name}"
        )
    ]

    if len(index_names) == 0:
        logging.info(f"Creating Vector Search index {index_name} ...")
        try:
            vs_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=VS_INDEX_NAME,
                dimensions=VS_DIMENSIONS,
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                shard_size="SHARD_SIZE_SMALL",
                index_update_method="STREAM_UPDATE",
                approximate_neighbors_count=APPROXIMATE_NEIGHBORS_COUNT,
            )
            logging.info(
                f"Vector Search index {vs_index.display_name} created with resource name {vs_index.resource_name}"
            )
            return vs_index
        except Exception as e:
            logging.error(f"Failed to create Vector Search index: {e}")
    else:
        vs_index = aiplatform.MatchingEngineIndex(index_name=index_names[0])
        logging.info(
            f"Vector Search index {vs_index.display_name} exists with resource name {vs_index.resource_name}"
        )
        return vs_index

def create_endpoint(
    endpoint_name:str, 
    public_endpoint_enabled:bool=False,
    enable_private_service_connect:bool=True
) -> aiplatform.MatchingEngineIndexEndpoint:
    """
        Creates a Vector Search index endpoint.

        Args:
        endpoint_name (str): The name of the index endpoint to be created.

        Returns:
        aiplatform.MatchingEngineIndexEndpoint: The created Vector Search index endpoint.
    """
    endpoint_names = [
        endpoint.resource_name
        for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f"display_name={endpoint_name}"
        )
    ]

    if len(endpoint_names) == 0:
        logging.info(
            f"Creating Vector Search index endpoint {endpoint_name} ..."
        )
        vs_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_name, 
            public_endpoint_enabled=public_endpoint_enabled,
            enable_private_service_connect=enable_private_service_connect
        )
        logging.info(
            f"Vector Search index endpoint {vs_endpoint.display_name} created with resource name {vs_endpoint.resource_name}"
        )
        return vs_endpoint
    else:
        vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_names[0]
        )
        logging.info(
            f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
        )
        return vs_endpoint

def deploy_index_at_endpoint(
    index:aiplatform.MatchingEngineIndex, 
    endpoint:aiplatform.MatchingEngineIndexEndpoint, 
    deployed_index_id:str, 
    display_name:str, 
    machine_type:str="e2-standard-16", 
    min_replica_count:int=1, 
    max_replica_count:int=1
) -> None:

    index_endpoints = [
        (deployed_index.index_endpoint, deployed_index.deployed_index_id)
        for deployed_index in index.deployed_indexes
    ]

    if len(index_endpoints) == 0:
        print(
            f"Deploying Vector Search index {index.display_name} at endpoint {endpoint.display_name} ..."
        )
        vs_deployed_index = endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id,
            display_name=display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
        )
        logging.info(
            f"Vector Search index {index.display_name} is deployed at endpoint {endpoint.display_name}"
        )
        pass
    else:
        vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoints[0][0]
        )
        logging.info(
            f"Vector Search index {index.display_name} is already deployed at endpoint {endpoint.display_name}"
        )
        pass

def setup_vector_store(
    project_id:str, 
    region:str, 
    index:aiplatform.MatchingEngineIndex, 
    endpoint:aiplatform.MatchingEngineIndexEndpoint, 
    gcs_bucket_name:str
) -> VertexAIVectorStore:
    # setup storage
    vector_store = VertexAIVectorStore(
        project_id=project_id,
        region=region,
        index_id=index.resource_name,
        endpoint_id=endpoint.resource_name,
        gcs_bucket_name=gcs_bucket_name,
    )

    return vector_store

def set_storage_context(
    vector_store:VertexAIVectorStore
) -> StorageContext:
    # set storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context

def set_embed_model(
    project_id:str, 
    region:str, 
    model_name:str="textembedding-gecko@003"
) -> VertexTextEmbedding:
    # configure embedding model
    embed_model = VertexTextEmbedding(
        model_name=model_name,
        project=project_id,
        location=region,
    )

    # setup the index/query process, ie the embedding model (and completion if used)
    Settings.embed_model = embed_model

    return embed_model

def add_nodes_to_vector_store(
    vector_store:VertexAIVectorStore, 
    text_list:List[str], 
    embed_model:VertexTextEmbedding
) -> None:
    nodes = [
        TextNode(text=text, embedding=embed_model.get_text_embedding(text))
        for text in text_list
    ]

    vector_store.add_nodes(nodes)





