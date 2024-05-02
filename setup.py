import setuptools

# set version
__version__ = "0.0.1"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-rag",
    version="0.0.1",
    author="Hieu Nguyen",
    author_email="hieunv.dev@gmail.com",
    description="A simple RAG pipeline for question answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/behitek/simple-rag/",
    project_urls={
        "Bug Tracker": "https://github.com/behitek/simple-rag/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["rag", "rag.*"]),
    modules=["rag"],
    python_requires=">=3.6",
    install_requires=[
        "sentence-transformers",
        "transformers",
        "requests",
        "google-generativeai",
        "rank_bm25",
        "pdftotext",
    ],
)
