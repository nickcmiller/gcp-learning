# Install Required Packages
pip install llama-index llama-index-vector-stores-vertex llama-index-llms-vertex

# Configure Vertex AI
from google.cloud import aiplatform

aiplatform.init(project="your_project_id", location="your_region")

# Create and Deploy an Index
from google.cloud import aiplatform

!gsutil mb -l your_region -p your_project_id gs://your_gcs_bucket

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="llamaindex-doc-index",
    gcs_source_uris=["gs://your_gcs_bucket"],
    dimensions=768,  # Adjust dimensions based on your embedding model
    index_update_method="BATCH_UPDATE"
)

endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="llamaindex-doc-endpoint"
)
index.deploy(endpoint=endpoint)

# Embed Documents Using LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.vertex import VertexAIEmbedding

Settings.embed_model = VertexAIEmbedding(model_name="text-embedding-004")

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(documents)

# Save Embeddings to Vertex AI
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

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
query_engine = index.as_query_engine()

response = query_engine.query("your query string")
print(response)