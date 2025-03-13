from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.core.indices import vector_store
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from qdrant_client import QdrantClient, AsyncQdrantClient
import tribore.utils as utils

client = QdrantClient(location=":memory:")
aclient = AsyncQdrantClient(location=":memory:")

llm = Ollama(
    model="llama3.2",
    temperature=0.3,
)
embedd_model = HuggingFaceEmbedding(
    model_name="jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True,
)
Settings.embed_model = embedd_model
Settings.llm = llm
Settings.chunk_size = 5120

parent_dir = Path(__file__).resolve().parents[1]
file1 = parent_dir /"data" / "all_sample.cm"
file2 = parent_dir /"data" / "all_sample.nl"
console = Console()


def format_panel_content(metadata: dict, text: str, score: float, additional_columns: list = None) -> str:
    """
    Formats the panel content with CLI id, text, score and any additional metadata.
    """
    content = [
        f"[cyan]CLI:[/] {text}",
        f"[yellow]Fusion Score:[/] {score:.4f}",
    ]
    if additional_columns:
        for col in additional_columns:
            value = metadata.get(col, "N/A")
            content.append(f"[green]{col.replace('_', ' ').capitalize()}:[/] {value}")
    return "\n".join(content)

documents = utils.load_line_by_line_with_metadata(file1, file2)

vector_store = QdrantVectorStore(
    collection_name="cli_strings",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    batch_size=20,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

user_query = r"""top -bn1 | grep -i 'zombie'"""

retriever = index.as_retriever(similarity_top_k=5, sparse_top_k=10, vector_store_query_mode="hybrid")
response = retriever.retrieve(user_query)

query_panel = Panel(
    Text(user_query, style="green"),
    title="CLI Search",
    border_style="bold blue",
)
console.print(query_panel)

for idx, node in enumerate(response, start=1):
    content_text = getattr(node.node, "text", str(node))
    metadata = getattr(node, "metadata", {})
    score = getattr(node, "score", 0.0)
    panel_content = format_panel_content(metadata, content_text, score)
    panel = Panel(
        panel_content,
        title=f"Source Node Rank: {idx}",
        border_style="magenta",
    )
    console.print(panel)

