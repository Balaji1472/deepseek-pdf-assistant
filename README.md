# PDF Assistant with DeepSeek and Pinecone

This is a PDF question-answering assistant built using Streamlit, Ollama (DeepSeek R1), and Pinecone. Upload a research PDF and ask questions — responses are generated using document-specific context only.

## Features

* Upload any PDF and extract context-based answers
* Uses DeepSeek R1 for both embeddings and LLM
* Stores chunks in Pinecone vector DB for fast retrieval
* Each query is isolated to the uploaded document
* Simple Streamlit UI with chat history

## Setup Instructions

1. Clone the repository:

```bash
https://github.com/Balaji1472/deepseek-pdf-assistant.git
cd deepseek-pdf-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Pull the model using Ollama:

```bash
ollama pull deepseek-r1
```

4. Set your Pinecone API key in `.streamlit/secrets.toml`:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
```

5. Run the app:

```bash
streamlit run ragpdfassist.py
```

---

## Notes

* If Pinecone is unavailable, the app uses in-memory fallback.
* Chat history and performance metrics are shown in the UI.

## License

This project is open-source and available under the MIT License.
