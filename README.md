# RAG Book Query System

A simple Retrieval-Augmented Generation (RAG) pipeline built in Go.  
Users can query a book and get relevant text excerpts as sourced responses.

---

## 1. Setup and Run

### Requirements
- Ubuntu (tested on 22.04+)
- Go 1.25.3+
- Git, Curl, Bash

### Clone and Prepare

```bash
git clone https://github.com/Nicolas-hometask/TASK.git
cd rag-book
go mod tidy
```

### Download a Sample Book
Example: *Alice in Wonderland* from Project Gutenberg.

```bash
chmod +x scripts/download_book.sh
./scripts/download_book.sh 11
```

This saves the book as:
```
data/book.txt
```

### Run the API Server

```bash
go run ./cmd/server
```

Then in another terminal, query it:

```bash
curl -X POST http://localhost:8080/api/v1/query   -H "Content-Type: application/json"   -d '{"query":"Who is the White Rabbit?","top_k":3}'
```

---

## 2. Architecture and Design

```
rag-book/
├── cmd/
│   ├── server/       # REST API
│   ├── eval/         # Evaluation (F1, Precision, Recall)
│   └── optimize/     # Grid search optimizer
├── internal/
│   ├── rag/          # Core RAG pipeline
│   ├── store/        # In-memory vector store
│   ├── embeddings/   # Hash-based embedding model
│   └── eval/         # Evaluation logic
├── scripts/
│   ├── download_book.sh
│   └── grid_eval.sh
├── data/
│   └── book.txt
├── testdata/
│   └── eval_cases.json
└── go.mod
```

### Design Highlights
- **Go backend** — fast, simple, easy to run.
- **In-memory vector store** — no external DB needed.
- **Hash embeddings** — lightweight placeholder for semantic embeddings.
- **Configurable parameters** — `top_k`, `chunk_size`, `chunk_overlap`, and `cosine_threshold`.
- **Evaluation module** — computes Precision, Recall, and F1.
- **Optimizer** — performs grid search to find the best parameter set.

### Possible Future Work
- Replace hash embeddings with a real embedding model.
- Add persistent vector storage (SQLite + pgvector or Weaviate).
- Add reranker or hybrid retrieval (vector + keyword).

---

## 3. Evaluation and Results

### How to Run Evaluation
```bash
go run ./cmd/eval --top_k=3 --chunk_size=800 --chunk_overlap=200 --cosine_threshold=0.2
```

### Grid Search
Run the built-in grid search optimizer:
```bash
go run ./cmd/optimize
```

It automatically tests parameter combinations and prints the best F1 score.

### Example Results

| Run | top_k | chunk | overlap | threshold | Precision | Recall | F1 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 3 | 800 | 200 | 0.0 | 0.55 | 0.65 | 0.59 |
| 2 | 3 | 800 | 200 | 0.2 | 0.68 | 0.60 | 0.64 |
| 3 | 2 | 600 | 200 | 0.4 | 0.75 | 0.50 | 0.60 |

**Best configuration:**
```
top_k=3, chunk_size=800, overlap=200, cosine_threshold=0.2
F1 = 0.64
```

---

## 4. Summary

| Metric | Baseline | Tuned |
|:--|:--:|:--:|
| Precision | 0.55 | 0.68 |
| Recall | 0.65 | 0.60 |
| F1 | 0.59 | **0.64** |

### Key Improvements
- Added cosine threshold filtering → higher precision.
- Balanced chunking and overlap → better context coverage.
- Tuned `top_k` for optimal trade-off between recall and precision.

---

## 5. Quick Commands

| Purpose | Command |
|:--|:--|
| Run API | `go run ./cmd/server` |
| Query API | `curl -X POST ...` |
| Evaluate F1 | `go run ./cmd/eval` |
| Optimize parameters | `go run ./cmd/optimize` |

---

## 6. Notes

This implementation focuses on simplicity and reproducibility.  
It uses deterministic embeddings and an in-memory store to demonstrate the RAG concept end-to-end.  
Future work can replace these components with real semantic and scalable backends.
