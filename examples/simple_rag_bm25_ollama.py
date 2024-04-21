import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retrieval import BM25Retrieval
from rag.llm import OllamaLLM
from rag.rerank import CrossEncoderRerank
from rag.pipeline import SimpleRAGPipeline, Answer
from rag.data_helper import PDFReader
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

query = "What can Ollama do?"
response: Answer = pipeline.run(query)
print(response.answer)
