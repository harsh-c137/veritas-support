# ScriptChain Health: Multi-Source RAG for Technical Support

## Approach & Architecture Overview
This repository contains a locally executable, Multi-Source RAG system designed to intelligently retrieve, weigh, and synthesize technical support answers for a fictional product ("Nexus Agent"). To ensure privacy, eliminate API latency, and maintain a lightweight footprint, the system relies entirely on local, open-source embedding and reranking models, paired with Llama-3.1-8B-Instruct for final generation.

### Requirement 1: Data Sourcing
Instead of scraping disparate and uncontrollable live web data, I engineered a highly controlled synthetic dataset using JSON. This allowed for precise injection of metadata (`source_type`, `title`, `author`) before the text hit the chunker. The data includes:
* **Documentation (`doc`):** Formal, factual infrastructure and API guidelines.
* **Technical Blogs (`blog`):** Opinionated developer workarounds and migration guides.
* **Customer Forums (`forum`):** Messy, user-generated Q&A, occasionally containing bad advice or hallucinations to test the system's contradiction resolution.

### Requirement 2: Chunking Strategy
I implemented source-specific chunking logic to preserve semantic integrity:
* **Documentation:** Processed using a `RecursiveCharacterTextSplitter` (chunk_size=400, overlap=50). Official docs are dense and hierarchical, requiring standard overlap to prevent cutting off technical steps.
* **Blogs & Forums:** Processed as whole, isolated chunks. Splitting a short forum reply or a concise blog post destroys the conversational context, so they are embedded intact.

### Requirement 3: Intelligent Weighting & Combination
A standard vector search often suffers from "keyword dominance," where forum posts with exact keyword matches crowd out official documentation. To intelligently combine sources, I implemented **Parallel Metadata Retrieval**. 
The vector store (Chroma) executes three simultaneous sub-searches filtered by `source_type`, pulling the top 2 docs, top 2 blogs, and top 2 forums. This guarantees the retrieval pool always maintains a 360-degree, weighted view of the knowledge base before reranking.

### Requirement 4: Reranking Mechanism
To bridge the gap between general semantic similarity (the bi-encoder) and deep logical relevance, I implemented a Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). 
The initial 6 chunks from the parallel retrieval are paired individually with the user's query and scored. The top 3 chunks are then selected. During testing, this successfully filtered out irrelevant chunks (like AWS billing blogs) and pushed highly specific, logically relevant forum posts to the top.

### Requirement 5: Contradiction Handling
The dataset was intentionally seeded with contradictory advice (e.g., a blog post suggesting a React workaround, while a forum post suggests downgrading the React version entirely). 
Rather than building brittle, hard-coded logic to detect semantic conflicts, I handled this via **System Prompt Hierarchy**. The LLM (Llama-3.1) is fed the context with explicit `source_type` tags attached to each chunk. The prompt enforces a strict truth hierarchy: `doc` > `blog` > `forum`. If a conflict is detected, the LLM is instructed to explicitly flag the contradiction to the user and resolve it based on this hierarchy.

### Requirement 6: Source Logging
Python's standard `logging` module is integrated into the generation step. Before invoking the LLM, the system logs the exact `source_type` metadata of the final, reranked chunks being passed into the context window, providing full observability into the RAG pipeline's final payload.

## Performance Analysis
* **Retrieval (BAAI/bge-small-en-v1.5):** By using a sub-150M parameter embedding model, the initial parallel vector retrieval operates in milliseconds on a standard CPU while maintaining top-tier MTEB benchmark performance.
* **Reranking (ms-marco-MiniLM-L-6-v2):** The cross-encoder adds marginal latency but massively increases the precision of the context window. It effectively acts as a high-quality filter, ensuring the LLM is only spending compute cycles on deeply relevant text, preventing hallucinations.
* **Contradiction Resolution:** The Llama-3.1 prompt engineering proved highly effective. In testing, it successfully identified conflicting React 18 advice between a forum and a blog, and correctly informed the user that it was prioritizing the blog's solution based on the established source hierarchy.