from google.cloud import aiplatform

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

def create_endpoint(endpoint_name:str) -> aiplatform.MatchingEngineIndexEndpoint:
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
            display_name=VS_INDEX_ENDPOINT_NAME, public_endpoint_enabled=True
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
):
    # check if endpoint exists
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
            display_name=VS_INDEX_NAME,
            machine_type="e2-standard-16",
            min_replica_count=1,
            max_replica_count=1,
        )
        logging.info(
            f"Vector Search index {index.display_name} is deployed at endpoint {endpoint.display_name}"
        )

        return vs_deployed_index
    else:
        vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoints[0][0]
        )
        logging.info(
            f"Vector Search index {index.display_name} is already deployed at endpoint {endpoint.display_name}"
        )

        return vs_deployed_index

