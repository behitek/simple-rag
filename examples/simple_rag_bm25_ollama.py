import os

from rag.data_helper import PDFReader
from rag.llm import OllamaLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import CrossEncoderRerank
from rag.retrieval import BM25Retrieval
from rag.text_utils import text2chunk

sample_pdf = os.path.join(os.path.dirname(__file__), "sample.pdf")
contents = PDFReader(pdf_paths=[sample_pdf]).read()
text = " ".join(contents)
chunks = text2chunk(text, chunk_size=200, overlap=50)
print(f"Number of chunks: {len(chunks)}")

retrieval = BM25Retrieval(documents=chunks)
llm = OllamaLLM(model_name="llama3:instruct")
rerank = CrossEncoderRerank(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
pipeline = SimpleRAGPipeline(retrieval=retrieval, llm=llm, rerank=rerank)


def run(query: str) -> Answer:
    return pipeline.run(query)


if __name__ == "__main__":
    query = "What can Ollama do?"
    print("Sample query:", query)
    response: Answer = pipeline.run(query)
    print(response.answer)
    print("Now, please ask your own questions!")
    while True:
        query = input("Your question: ")
        response: Answer = run(query)
        print(response.answer)
        print()
