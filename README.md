AI-Powered PDF Question Answering with RAG

This application lets you upload one or multiple PDF files, transform their content into embeddings, and store them inside a long-term vector database. You can then ask natural language queries, and the system retrieves and generates answers strictly from your uploaded PDFs — ensuring precise, context-aware responses.

🔑 Highlights

Multiple PDF Support — Upload and process several documents together.

Optimized Text Handling — Extracts content with PyMuPDF, splits into sentence/paragraph chunks with overlap for better retrieval.

Persistent Vector Storage — Uses ChromaDB to manage embeddings efficiently.

Compact Embeddings — Built on Sentence Transformers (MiniLM) for high-quality vector representations.

Answer Generation

Integrates with OpenAI GPT models (when OPENAI_API_KEY is provided).

Provides fallback with HuggingFace Transformers for local inference.

Traceable Responses — Each answer includes source PDF, page reference, and text snippet.

User Interfaces

Web App: Streamlit-based, chat-style Q&A.

CLI: Command-line support for both ingestion and querying.

🛠️ Technology Stack

Frontend/UI: Streamlit

PDF Text Extraction: PyMuPDF

Embeddings: Sentence Transformers

Vector Database: ChromaDB

LLMs: OpenAI API (optional) / HuggingFace
