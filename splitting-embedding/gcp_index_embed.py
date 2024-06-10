# Docs: https://docs.llamaindex.ai/en/stable/examples/vector_stores/VertexAIVectorSearchDemo/

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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import QueryEngine
from llama_index.data_structs.node import Node
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.docstore import DocumentStore, Document
from llama_index.storage.index_store import IndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import VectorStore


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

# GCP functions

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
    """
        Deploys a Vector Search index at an endpoint.

        Args:
        index (aiplatform.MatchingEngineIndex): The index to be deployed.
        endpoint (aiplatform.MatchingEngineIndexEndpoint): The endpoint to deploy the index at.
        deployed_index_id (str): The ID of the deployed index.
        display_name (str): The name of the deployed index.
        machine_type (str): The machine type to deploy the index at.
        min_replica_count (int): The minimum number of replicas to deploy the index at.
        max_replica_count (int): The maximum number of replicas to deploy the index at.
    """
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

# LlamaIndex functions

def setup_vector_store(
    project_id:str, 
    region:str, 
    index:aiplatform.MatchingEngineIndex, 
    endpoint:aiplatform.MatchingEngineIndexEndpoint, 
    gcs_bucket_name:str
) -> VertexAIVectorStore:
    """
        Setups a Vector Store.

        Args:
        project_id (str): The project ID.
        region (str): The region.
        index (aiplatform.MatchingEngineIndex): The index to be deployed.
        endpoint (aiplatform.MatchingEngineIndexEndpoint): The endpoint to deploy the index at.
        gcs_bucket_name (str): The GCS bucket name.

        Returns:
        VertexAIVectorStore: The Vector Store.
    """
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
    """
        Setups a Storage Context.

        Args:
        vector_store (VertexAIVectorStore): The Vector Store.

        Returns:
        StorageContext: The Storage Context.
    """
    # set storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context

def set_embed_model(
    project_id:str, 
    region:str, 
    model_name:str="textembedding-gecko@003"
) -> VertexTextEmbedding:
    """
        Setups an embedding model.

        Args:
        project_id (str): The project ID.
        region (str): The region.
        model_name (str): The model name.

        Returns:
        VertexTextEmbedding: The embedding model.
    """
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
    """
        Adds nodes to a Vector Store.

        Args:
        vector_store (VertexAIVectorStore): The Vector Store.
        text_list (List[str]): The list of texts.
        embed_model (VertexTextEmbedding): The embedding model.
    """
    nodes = [
        TextNode(text=text, embedding=embed_model.get_text_embedding(text))
        for text in text_list
    ]

    try:
        vector_store.add_nodes(nodes)
        logging.info(f"Added {len(nodes)} nodes to vector store")
    except Exception as e:
        logging.error(f"Failed to add records to vector store: {e}")
    pass

def create_retriever(
    vector_store:VertexAIVectorStore, 
    embed_model:VertexTextEmbedding
) -> VectorIndexRetriever:
    """
        Creates a retriever.

        Args:
        vector_store (VertexAIVectorStore): The Vector Store.
        embed_model (VertexTextEmbedding): The embedding model.

        Returns:
        VectorIndexRetriever: The retriever.
    """
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    retriever = index.as_retriever()

    return retriever

##############
 # Examples #
##############

# Example 1: Add documents with metadata attributes and use filters¶

def add_records_to_vector_store_with_metadata(
    vector_store:VertexAIVectorStore, 
    embed_model:VertexTextEmbedding, 
    dict_list:List[Dict[str, str]],
    embed_field:str
) -> None:
    """
        Adds records to a Vector Store with metadata.

        Args:
        vector_store (VertexAIVectorStore): The Vector Store.
        embed_model (VertexTextEmbedding): The embedding model.
        dict_list (List[Dict[str, str]]): The list of dictionaries.
        embed_field (str): The field to embed.
    """
    nodes = []
    for d in dict_list:
        text = d.pop(embed_field)
        embedding = embed_model.get_text_embedding(text)
        metadata = {**d}
        nodes.append(TextNode(text=text, embedding=embedding, metadata=metadata))

    try:
        vector_store.add(nodes)
        logging.info(f"Added {len(records)} records with metadata to vector store")
    except Exception as e:
        logging.error(f"Failed to add records to vector store: {e}")
    pass
"""
records = [
     {
         "description": "A versatile pair of dark-wash denim jeans."
         "Made from durable cotton with a classic straight-leg cut, these jeans"
         " transition easily from casual days to dressier occasions.",
         "price": 65.00,
         "color": "blue",
         "season": ["fall", "winter", "spring"],
     },
     {
         "description": "A lightweight linen button-down shirt in a crisp white."
         " Perfect for keeping cool with breathable fabric and a relaxed fit.",
         "price": 34.99,
         "color": "white",
         "season": ["summer", "spring"],
     },
     {
         "description": "A soft, chunky knit sweater in a vibrant forest green. "
         "The oversized fit and cozy wool blend make this ideal for staying warm "
         "when the temperature drops.",
         "price": 89.99,
         "color": "green",
         "season": ["fall", "winter"],
    },
]

index = create_index(VS_INDEX_NAME, VS_DIMENSIONS, "DOT_PRODUCT_DISTANCE", "SHARD_SIZE_SMALL", "STREAM_UPDATE", APPROXIMATE_NEIGHBORS_COUNT)

endpoint = create_endpoint(VS_INDEX_ENDPOINT_NAME)

vector_store = setup_vector_store(PROJECT_ID, REGION, index, endpoint, GCS_BUCKET_NAME)

embed_model = set_embed_model(PROJECT_ID, REGION)

add_records_to_vector_store_with_embedding(vector_store, embed_model, records, "description")
"""

def similarity_search_without_filters(
    vector_store:VertexAIVectorStore, 
    embed_model:VertexTextEmbedding,
    query:str
) -> List[Document]:
    """
        Performs a similarity search without filters.

        Args:
        vector_store (VertexAIVectorStore): The Vector Store.
        embed_model (VertexTextEmbedding): The embedding model.
        query (str): The query.

        Returns:
        List[Document]: The list of documents.
    """
    retriever = create_retriever(vector_store, embed_model)
    response = retriever.retrieve(query)

    return response

"""
response = similarity_search_without_filters(vector_store, embed_model, "pants")
for row in response:
    print(f"Text: {row.get_text()}")
    print(f"   Score: {row.get_score():.3f}")
    print(f"   Metadata: {row.metadata}")
"""

def similarity_search_with_filters(
    vector_store:VertexAIVectorStore, 
    embed_model:VertexTextEmbedding,
    query:str,
    filters:List[MetadataFilter]
) -> None:
    retriever = create_retriever(vector_store, embed_model)

    response = retriever.retrieve(
        query=query,
        filters=filters
    )

    return response

"""
filters=[
     MetadataFilter(key="color", value="blue"),
     MetadataFilter(key="price", operator=FilterOperator.GT, value=70.0),
 ]
 similarity_search_with_filters(vector_store, embed_model, "pants", filters)
 """

# Example 2: Parse, Index and Query PDFs using Vertex AI Vector Search and Gemini Pro¶

def create_query_engine(
    documents:List[Document],
    vector_store:VertexAIVectorStore,
    storage_context:StorageContext
) -> QueryEngine:

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    query_engine = index.as_query_engine()

    return query_engine

"""
! mkdir -p ./data/arxiv/
! wget 'https://arxiv.org/pdf/1706.03762.pdf' -O ./data/arxiv/test.pdf
data_path = "./data/arxiv"

vertex_gemini = Vertex(model="gemini-pro", temperature=0, additional_kwargs={})
Settings.llm = vertex_gemini

embed_model = set_embed_model(PROJECT_ID, REGION)
Settings.embed_model = embed_model

index = create_index(VS_INDEX_NAME, VS_DIMENSIONS, "DOT_PRODUCT_DISTANCE", "SHARD_SIZE_SMALL", "STREAM_UPDATE", APPROXIMATE_NEIGHBORS_COUNT)

endpoint = create_endpoint(VS_INDEX_ENDPOINT_NAME)

vector_store = setup_vector_store(PROJECT_ID, REGION, index, endpoint, GCS_BUCKET_NAME)
storage_context = set_storage_context(vector_store)

documents = SimpleDirectoryReader(data_path).load_data()

query_engine = create_query_engine(documents, vector_store, storage_context)

response = query_engine.query(
    "who are the authors of paper Attention is All you need?"
)

print(f"Response:")
print("-" * 80)
print(response.response)
print("-" * 80)
print(f"Source Documents:")
print("-" * 80)
for source in response.source_nodes:
    print(f"Sample Text: {source.text[:50]}")
    print(f"Relevance score: {source.get_score():.3f}")
    print(f"File Name: {source.metadata.get('file_name')}")
    print(f"Page #: {source.metadata.get('page_label')}")
    print(f"File Path: {source.metadata.get('file_path')}")
    print("-" * 80)

"""


