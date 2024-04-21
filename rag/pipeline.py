from abc import ABC
from .retrieval import BaseRetrieval
from .rerank import BaseRerank
from .llm import BaseLLM
from .prompt import ANSWER_PROMPT


class Answer:
    def __init__(self, answer: str, contexts: list[str]):
        self.answer = answer
        self.contexts = contexts


class Pipeline(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, query: str) -> Answer:
        raise NotImplementedError


class SimpleRAGPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("retrieval"):
            raise ValueError("Please provide a `retrieval` model.")
        if not kwargs.get("llm"):
            raise ValueError("Please provide a `llm` model.")
        self.retrieval = kwargs.get("retrieval")
        # assert retrieval must be BaseRetrieval class or it inherits from BaseRetrieval
        assert issubclass(self.retrieval.__class__, BaseRetrieval)
        self.rerank = kwargs.get("rerank")
        if self.rerank:
            # assert rerank must be BaseRerank class or it inherits from BaseRerank
            assert issubclass(self.rerank.__class__, BaseRerank)
        self.llm = kwargs.get("llm")
        # assert llm must be BaseLLM class or it inherits from BaseLLM
        assert issubclass(self.llm.__class__, BaseLLM)
        self.retrieval_top_k = kwargs.get("retrieval_top_k", 100)
        self.rerank_top_k = kwargs.get("rerank_top_k", 3)

    def run(self, query: str) -> Answer:
        # Retrieve documents
        relevant_docs, relevant_meta = self.retrieval.retrieve(
            query, top_k=self.retrieval_top_k
        )
        # Rerank documents
        if self.rerank:
            reranked_docs, scores = self.rerank.rerank(
                query, relevant_docs, top_k=self.rerank_top_k
            )
        else:
            reranked_docs = relevant_docs
        # Generate answer
        prompt = ANSWER_PROMPT.format(query=query, context="\n".join(reranked_docs))
        answer = self.llm.generate(prompt)
        return Answer(answer=answer, contexts=reranked_docs)
