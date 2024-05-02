![](/assets/flow.png)
# Simple RAG
This is a simple RAG (Retrieval-Augmented Generation) that mostly self-implemented. This simple-rag package contain 4 modules:
- **Retrieval**: A retriever that retrieve the most relevant documents from a given corpus.
- **Rerank**: A reranker that rerank the retrieved documents.
- **LLM**: A language model that generate the answer.
- **Data Helper**: A helper that help to load the PDF data.

## Installation

**Pre-requisites**:
- Python 3.6 or later
- Ollama (for LLM self-hosted)
- Poppler (for PDF processing)

To install poppler, select one of the following commands that is appropriate for your OS:
```bash
# Debian/Ubuntu
sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev

# Fedora/RHEL
sudo yum install gcc-c++ pkgconfig poppler-cpp-devel python3-devel

# macOS
brew install pkg-config poppler python

# Windows (using conda)
conda install -c conda-forge poppler
```

Then, install the package using the following commands:
```bash
git clone https://github.com/behitek/simple-rag/
cd simple-rag
pip install -e .
```

## How to use
Here is an [example](/examples/simple_rag_bm25_ollama.py) of how to use the simple-rag package:
```python
import os

from rag.data_helper import PDFReader
from rag.llm import OllamaLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import CrossEncoderRerank
from rag.retrieval import BM25Retrieval
from rag.text_utils import text2chunk

# Set your PDF path here
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
```