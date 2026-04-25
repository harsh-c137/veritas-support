import os
from app import (
    ingest_and_chunk,
    build_vectorstore,
    retrieve_and_combine,
    rerank_results,
    generate_answer,
)

QUERIES = [
    "How do I fix the React 18 strict mode error with NexusProvider?",
    "What is the recommended temperature setting to reduce hallucinations in Nexus Agent?",
    "How do I deploy the Nexus Agent on AWS?",
    "What IAM permissions are required for the nexus-cli init command?",
    "How can I avoid unexpected AWS costs when testing Nexus Agent locally?",
    "How do I embed the Nexus Agent chat widget in a React app?",
    "What is the maximum chunk size for the Nexus Agent retrieval engine?",
    "Is there a local-only mode for Nexus Agent that bypasses cloud provisioning?",
    "Why is the Nexus Agent making up fake API endpoints?",
    "What TypeScript packages do I need to install for Nexus Agent frontend integration?",
]

OUTPUT_FILE = "QA_Results.md"

def main():
    docs = ingest_and_chunk()
    vs = build_vectorstore(docs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Multi-Source RAG - Example Queries & Responses\n\n")

        for idx, query in enumerate(QUERIES, start=1):
            retrieved = retrieve_and_combine(query, vs)
            reranked = rerank_results(query, retrieved, top_k=3)
            sources = [doc.metadata.get("source_type", "unknown") for doc, _ in reranked]
            answer = generate_answer(query, reranked)

            f.write(f"### Query {idx}: {query}\n\n")
            f.write(f"**Sources Retrieved:** `{sources}`\n\n")
            f.write(f"**System Response:**\n\n")
            for line in answer.strip().splitlines():
                f.write(f"> {line}\n")
            f.write("\n---\n\n")

if __name__ == "__main__":
    main()
