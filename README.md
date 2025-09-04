# arXiv RAG Assistant

A Retrieval-Augmented Generation (RAG) system for arXiv papers that provides intelligent paper search and Q&A capabilities.

## 🌟 Features

- **Smart Paper Search**: Advanced retrieval system for arXiv papers
- **Context-Aware Responses**: Uses RAG to provide accurate, context-based answers
- **FastAPI Backend**: Robust API implementation for retrieval and generation
- **Streamlit UI**: User-friendly interface for interacting with the system
- **Vector Database**: Efficient document storage and similarity search
- **Reranking**: Enhanced retrieval accuracy through reranking

## 🛠️ Project Structure

```
arxiv-rag-assistant/
├── api/                 # FastAPI backend implementation
├── configs/            # Configuration files
├── data/              # Data storage
│   ├── chunks/        # Chunked paper data
│   ├── processed/     # Processed data
│   ├── raw/          # Raw paper data
│   └── chroma/       # Vector database
├── notebooks/         # Jupyter notebooks for experiments
├── reports/          # Analysis reports and findings
├── src/              # Source code
│   ├── embeddings/   # Embedding generation
│   ├── evaluation/   # System evaluation
│   ├── generator/    # Text generation
│   ├── ingestion/    # Data ingestion
│   ├── pipeline/     # RAG pipeline
│   ├── reranker/     # Response reranking
│   └── vectordb/     # Vector database operations
├── tests/            # Test suites
└── ui/               # Streamlit frontend
```

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   python -m venv arxiv-rag-env
   source arxiv-rag-env/bin/activate  # On Windows: .\arxiv-rag-env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure the System**
   - Update `configs/config.yaml` with your settings
   - Configure model parameters in `configs/prompts.yaml`

3. **Run the Application**
   ```bash
   # Start the API server
   uvicorn api.main:app --reload --port 8000

   # Launch the UI
   streamlit run ui/app.py
   ```

## 🔧 API Endpoints

- `POST /api/retrieve`: Retrieve relevant contexts for a query
  ```json
  {
    "query": "string",
    "top_k": "int",
    "rerank_top_n": "int",
    "token_budget": "int"
  }
  ```

- `GET /api/health`: Health check endpoint

## 🎯 Components

1. **Data Ingestion**
   - Downloads papers from arXiv
   - Processes and chunks documents
   - Generates embeddings

2. **Retrieval System**
   - Vector similarity search
   - Reranking for improved relevance
   - Context selection based on token budget

3. **User Interface**
   - Paper search functionality
   - Q&A interface
   - Result visualization

## 📝 Usage Examples

1. **Simple Query**
   ```python
   import requests

   response = requests.post(
       "http://localhost:8000/api/retrieve",
       json={
           "query": "Latest developments in transformer architecture",
           "top_k": 5
       }
   )
   results = response.json()
   ```

2. **Using the UI**
   - Navigate to `http://localhost:8501`
   - Enter your query in the search box
   - View retrieved papers and ask follow-up questions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- arXiv for providing access to research papers
- Hugging Face for transformer models
- ChromaDB for vector storage
