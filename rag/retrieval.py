from abc import ABC
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi


class BaseRetrieval(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def ingest(
        self, documents: list[str], metadata: Optional[list[dict]] = None
    ) -> bool:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        # Some retrieval models might not need reranking
        raise NotImplementedError("This retrieval model does not support reranking.")


class BM25Retrieval(BaseRetrieval):

    def __init__(self, *args, **kwargs):
        """Initialize the BM25 retrieval model.
        - documents: list[str]: The list of documents to ingest.
        - metadata: list[dict]: The metadata associated with the documents.
        """
        super().__init__(*args, **kwargs)
        # Initialize BM25 model here
        documents = kwargs.get("documents")
        metadata = kwargs.get("metadata")
        if not documents:
            raise ValueError("Please provide a list of documents.")
        self.bm25 = None
        self.documents = None
        self.metadata = None
        self.__ingest__(documents, metadata)

    def __ingest__(self, documents: list[str], metadata=None):
        """Ingest the documents to the BM25 model.
        - documents: list[str]: The list of documents to ingest.
        - metadata: list[dict]: The metadata associated with the documents.
        """
        assert len(documents) > 0, "List of documents is empty."
        if metadata is not None:
            assert len(documents) == len(
                metadata
            ), "Length of metadata should be same as length of documents."
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.metadata = metadata
        self.documents = documents
        return True

    def retrieve(self, query: str, top_k: int = 10):
        """Retrieve the top-k documents based on the query.
        - query: str: The query to retrieve the documents with.
        - top_k: int: The number of documents to return.
        """
        if not self.bm25:
            raise RuntimeError(
                "BM25 model has not been initialized. Call ingest() first."
            )
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        metadata = None
        if self.metadata:
            metadata = [self.metadata[i] for i in top_n]
        documents = [self.documents[i] for i in top_n]
        return documents, metadata
