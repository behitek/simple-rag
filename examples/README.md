## How to use
To run these examples, you need to install the package `simple-rag` first. You can install the package follow the instructions in the project's README.md.


## CLI Example
The CLI example is currently ingesting a PDF file and answer the initial query. The user can then ask their own questions.

```bash
python examples/simple_rag_bm25_ollama.py
```

## Streamlit Example (Web App)

The Streamlit example is a simple web app that allows the user to input PDF file to ingest and ask questions.

```bash
streamlit run examples/simple_rag_bm25_ollama_ui.py
```

![](/assets/ui.png)