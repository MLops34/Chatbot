Here is a clean, professional, and well-structured **README.md** file based on the exact project structure and files you shared:

```markdown
ChatBot

RAG-powered chatbot for querying personalized Pdfs (December 2025 data) using Excel → JSON → FAISS → LLM.

## Project Structure


## Quick Start (Local)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Set OpenRouter API key
:

```toml
OPENROUTER_API_KEY = "--"
```

Or use environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

3. Ingest data (for each fund)

```bash
# Step 1: Excel sheet → JSON documents
python Fetch-01.py

# Step 2: (optional) Chunk
python Chunk-02.py

# Step 3: Embed & add to FAISS
python Embed-03.py
```

4. Run the main chatbot

```bash
streamlit run DecisionNode.py
```

Alternative versions:

```bash
streamlit run inter-04.py
streamlit run app.py
```

Open → http://localhost:8501

## Docker Deployment

### Build

```bash
docker build -t finance-rag .
```

### Run (recommended – mount persistent folders)

```bash
docker run -d \
  --name finance-rag \
  -p 8501:8501 \
  -v "$(pwd)/db/faiss_motilal:/app/db/faiss_motilal" \
  -v "$(pwd)/temp_pickles:/app/temp_pickles" \
  -e OPENROUTER_API_KEY="your-key-here" \
  finance-rag
```

Or use docker-compose:

```bash
docker-compose up -d
```


## Troubleshooting

- **White screen** → check terminal for traceback  
- **No funds shown** → run ingestion pipeline first  
- **Wrong fund retrieved** → add fund prefix (see above)  
- **LLM fails** → verify `OPENROUTER_API_KEY` and quota  
- **FAISS load error** → delete `db/faiss_motilal/` and re-embed

## License

MIT
```

Feel free to copy-paste this directly into your `README.md`.

If you want any section expanded (e.g. more detailed Docker instructions, contribution guide, badges, screenshots section, etc.), just tell me. 😄
