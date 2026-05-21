from retrieval.retriever import get_hybrid_retriever
from orchestration.tools import preload_retrievers
import os

preload_retrievers()
hybrid = get_hybrid_retriever(use_reranker=False)
docs = hybrid.invoke("Bronze plan deductible")
for i, d in enumerate(docs[:5]):
    print(f"--- DOC {i} ---")
    print(f"Source: {d.metadata.get('source_file')}")
    print(f"Content:\n{d.page_content}")
