from abc import ABC
from sentence_transformers import CrossEncoder


class BaseRerank(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> tuple[list, list]:
        raise NotImplementedError


class CrossEncoderRerank(BaseRerank):
    def __init__(self, *args, **kwargs):
        """Initialize the Cross-Encoder reranker.
        - model_name: str: The name of the model to use."""
        super().__init__(*args, **kwargs)
        if "model_name" not in kwargs:
            raise ValueError("Please provide a model_name.")
        self.model = CrossEncoder(kwargs["model_name"])

    def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> tuple[list, list]:
        """Rerank the documents based on the query.
        - query: str: The query to rerank the documents with.
        - documents: list[str]: The documents to rerank.
        - top_k: int: The number of documents to return.
        """
        cross_inp = [[query, passage] for passage in documents]
        cross_scores = self.model.predict(cross_inp)
        passage_scores = {}
        for idx, score in enumerate(cross_scores):
            passage_scores[idx] = score
        # sorted by cross-encoder scores desc
        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True
        )
        relevants = []
        scores = []
        for idx, score in sorted_passages[:top_k]:
            if score > 0:
                relevants.append(documents[idx])
                scores.append(score)
        if len(relevants) == 0:
            idx, score = sorted_passages[0]
            relevants.append(documents[idx])
            scores.append(score)
        return relevants, scores
