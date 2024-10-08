import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document
import asyncio

class EnhancedRetriever:
    def __init__(self, vectorstore, top_k=5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    async def get_relevant_documents(self, query):
        retriever = self.vectorstore.as_retriever()
        documents = await asyncio.to_thread(retriever.invoke, query)
        # Filter and rank the results using BM25
        ranked_documents = self.filter_and_rank_bm25(documents, query)
        return ranked_documents[:self.top_k]

    def filter_and_rank_bm25(self, documents, query):
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        if not texts:
            return []

        # Use BM25 to rank documents
        tokenized_texts = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]

        # Rank documents
        ranked_documents = [documents[idx] for idx in ranked_indices]
        return ranked_documents
