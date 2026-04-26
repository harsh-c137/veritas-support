import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_json(filename: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        return json.load(f)

def ingest_and_chunk() -> list[Document]:
    documents: list[Document] = []

    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    for entry in load_json("docs.json"):
        splits = doc_splitter.create_documents(
            texts=[entry["content"]],
            metadatas=[entry["metadata"]],
        )
        documents.extend(splits)

    for entry in load_json("blogs.json"):
        documents.append(
            Document(page_content=entry["content"], metadata=entry["metadata"])
        )

    for entry in load_json("forums.json"):
        documents.append(
            Document(page_content=entry["content"], metadata=entry["metadata"])
        )

    return documents

def build_vectorstore(documents: list[Document]) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    return vectorstore

def retrieve_and_combine(query: str, vectorstore: Chroma) -> list[Document]:
    source_configs = [
        {"source_type": "doc",   "k": 2},
        {"source_type": "blog",  "k": 2},
        {"source_type": "forum", "k": 2},
    ]

    combined_results: list[Document] = []

    for cfg in source_configs:
        results = vectorstore.similarity_search(
            query,
            k=cfg["k"],
            filter={"source_type": cfg["source_type"]},
        )
        combined_results.extend(results)

    return combined_results

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(
    query: str, documents: list[Document], top_k: int = 3
) -> list[tuple[Document, float]]:
    pairs = [[query, doc.page_content] for doc in documents]
    scores = _reranker.predict(pairs)

    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]

_llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=512,
    temperature=0.3,
)
_chat_llm = ChatHuggingFace(llm=_llm_endpoint)

SYSTEM_PROMPT = (
    "You are a support AI. Answer the user's query using ONLY the provided context. "
    "If the sources in the context contradict each other, you MUST resolve the conflict "
    "using this hierarchy: Official Documentation ('doc') is the absolute truth. "
    "Technical Blogs ('blog') are secondary. User Forums ('forum') are the least reliable. "
    "If you detect a contradiction, explicitly state that the sources conflict, briefly "
    "explain the conflict, and tell the user which source you are trusting based on the hierarchy."
)

def generate_answer(
    query: str, top_chunks: list[tuple[Document, float]]
) -> str:
    sources_used = [doc.metadata.get("source_type", "unknown") for doc, _ in top_chunks]
    logger.info(f"Sources used for this query: {sources_used}")

    context_parts: list[str] = []
    for i, (doc, score) in enumerate(top_chunks):
        src = doc.metadata.get("source_type", "unknown")
        title = doc.metadata.get("title", "N/A")
        context_parts.append(
            f"[Source {i+1} | type={src} | title={title}]\n{doc.page_content}"
        )
    context_block = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context_block}\n\n"
                f"Question: {query}"
            )
        ),
    ]

    logger.info("Invoking LLM...")
    response = _chat_llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    steps = [
        ("Loading data",         lambda: ingest_and_chunk()),
        ("Building vector store", None),
    ]

    with tqdm(total=3, desc="Initialising", ncols=60, bar_format="{l_bar}{bar}") as bar:
        docs = ingest_and_chunk()
        bar.update(1)
        bar.set_description("Building vector store")
        vs = build_vectorstore(docs)
        bar.update(1)
        bar.set_description("Ready")
        bar.update(1)

    query = input("\nEnter your question: ").strip()
    retrieved = retrieve_and_combine(query, vs)
    reranked = rerank_results(query, retrieved, top_k=3)
    answer = generate_answer(query, reranked)
    print(f"\n{answer}")
