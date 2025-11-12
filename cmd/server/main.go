package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"ragbook/internal/api"
	"ragbook/internal/embeddings"
	"ragbook/internal/rag"
	"ragbook/internal/store"
)

func mustReadBook(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("failed to read book file %s: %v", path, err)
	}
	return string(data)
}

func main() {
	bookPath := os.Getenv("BOOK_PATH")
	if bookPath == "" {
		bookPath = "data/book.txt"
	}
	bookID := os.Getenv("BOOK_ID")
	if bookID == "" {
		bookID = "detective-fiction"
	}

	log.Printf("Loading book from: %s (bookID=%s)", bookPath, bookID)

	bookText := mustReadBook(bookPath)

	embedder := embeddings.NewHashEmbedder(512)
	vectorStore := store.NewMemoryStore()
	pipeline := rag.NewPipeline(vectorStore, embedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	log.Printf("Ingesting book into vector store...")
	nChunks, err := pipeline.IngestBook(ctx, bookID, bookText, rag.IngestConfig{
		ChunkSize:       800,
		ChunkOverlap:    200,
		MaxChunks:       0,
		NormalizeSpaces: true,
	})
	if err != nil {
		log.Fatalf("failed to ingest book: %v", err)
	}
	log.Printf("Ingested %d chunks", nChunks)

	router := api.NewRouter(pipeline)
	addr := ":8080"
	log.Printf("Server started at %s", addr)
	if err := http.ListenAndServe(addr, router); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
