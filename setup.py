from setuptools import setup, find_packages

setup(
    name="ollama_pdf_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "langchain-text-splitters",
        "langchain-core",
        "pdfplumber",
        "ollama",
        "chromadb",
        "unstructured",
    ],
    python_requires=">=3.8",
) 