# A simple example of using the RAG-BM25-OLLAMA model with a Streamlit UI
# Allow ingesting a PDF file and asking questions
import os

import streamlit as st

from rag.data_helper import PDFReader
from rag.llm import OllamaLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import CrossEncoderRerank
from rag.retrieval import BM25Retrieval
from rag.text_utils import text2chunk

st.title("RAG-BM25-OLLAMA")
st.write("A simple example of using the RAG-BM25-OLLAMA model with a Streamlit UI")

# PDF Upload
pdf_path = os.path.join(os.path.dirname(__file__), "tmp.pdf")
st.header("Upload a PDF file")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("PDF file uploaded successfully!")
# Ingest PDF
if st.button("Ingest PDF"):
    contents = PDFReader(pdf_paths=[pdf_path]).read()
    text = " ".join(contents)
    chunks = text2chunk(text, chunk_size=200, overlap=50)
    st.write(f"Number of chunks: {len(chunks)}")

    retrieval = BM25Retrieval(documents=chunks)
    llm = OllamaLLM(model_name="llama3:instruct")
    rerank = CrossEncoderRerank(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    pipeline = SimpleRAGPipeline(retrieval=retrieval, llm=llm, rerank=rerank)
    st.write("PDF ingested successfully!")
    # save to session state
    st.session_state.pipeline = pipeline


# Ask questions visible only after ingesting PDF file
st.header("Ask Questions")
query = st.text_input("Your question:")
ingested_pipeline = st.session_state.get("pipeline")
if st.button("Ask") and ingested_pipeline:
    response: Answer = ingested_pipeline.run(query)
    st.write(response.answer)
elif not ingested_pipeline:
    st.write("Please ingest a PDF file first.")
