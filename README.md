Here is a clean, professional, and well-structured **README.md** file based on the exact project structure and files you shared:

```markdown
# FinanceRagChatBot

RAG-powered chatbot for querying **Motilal Oswal mutual fund portfolio holdings** (December 2025 data) using Excel → JSON → FAISS → LLM.

## Project Structure

```text
FinanceRagChatBot/
├── data/
│   └── raw/
│       └── db566-scheme-portfolio-details-december-2025.xlsx   # Source portfolio Excel file
├── db/
│   └── faiss_motilal/                                          # Persistent FAISS vector index
├── temp_pickles/                                               # Intermediate JSON files from ingestion
├── .gitattributes                                              # Git configuration
├── .gitignore                                                  # Git ignore patterns
├── app.py                                                      # Legacy / alternative Streamlit app
├── Chunk-02.py                                                 # Document chunking script
├── DecisionNode.py                                             # Main Streamlit chatbot + decision node
├── Dockerfile                                                  # Docker build configuration
├── docker-compose.yml                                          # Docker Compose setup
├── Embed-03.py                                                 # Embed documents & update FAISS index
├── Fetch-01.py                                                 # Excel → JSON documents (ingestion step 1)
├── inter-04.py                                                 # Alternative chatbot version with decision logic
├── README.md                                                   # This documentation
├── requirements.txt                                            # Python dependencies
├── Retrieval.py                                                # Utility to inspect retrieved chunks
└── AI Financial Analyst Agent (RAG-BASED LLM).pptx             # Project presentation overview
```

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

## Recommended Improvement (strongly suggested)

Add fund identifier prefix in `Fetch-01.py` → significantly improves retrieval accuracy

```python
# Inside load_sheet_to_documents(), when building row_content:
prefix = f"[FUND: {fund_name} ({sheet_name.upper()})] "
row_content = prefix + " • ".join(cells)
```

After adding → **re-run Fetch → Chunk → Embed** for all funds.

## Main Scripts Summary

| Script              | Purpose                                      | When to run                  |
|---------------------|----------------------------------------------|------------------------------|
| `Fetch-01.py`       | Excel sheet → JSON documents                 | Every new fund               |
| `Chunk-02.py`       | Optional: split long rows into chunks        | If documents are very long   |
| `Embed-03.py`       | Embed JSON → update/add to FAISS index       | After fetch/chunk            |
| `DecisionNode.py`   | Main chatbot + smart fund routing            | Most complete version        |
| `inter-04.py`       | Alternative chatbot implementation           | Compare / fallback           |
| `app.py`            | Legacy / simpler version                     | Rarely needed                |
| `Retrieval.py`      | Debug tool: see exactly what was retrieved   | Development / troubleshooting|

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
