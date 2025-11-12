package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"ragbook/internal/embeddings"
	"ragbook/internal/eval"
	"ragbook/internal/rag"
	"ragbook/internal/store"
)

func main() {
	// ---- Flags ----
	bookPath := flag.String("book", "data/book.txt", "Path to book text file")
	bookID := flag.String("id", "alice-in-wonderland", "Book ID label")
	evalFile := flag.String("eval", "testdata/eval_cases.json", "Path to evaluation cases JSON")
	topK := flag.Int("top_k", 3, "Number of chunks to retrieve per query")
	flag.Parse()

	// ---- Components ----
	embedder := embeddings.NewHashEmbedder(512)
	vectorStore := store.NewMemoryStore()
	pipeline := rag.NewPipeline(vectorStore, embedder)

	text, err := os.ReadFile(*bookPath)
	if err != nil {
		log.Fatalf("read book: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	_, err = pipeline.IngestBook(ctx, *bookID, string(text), rag.IngestConfig{
		ChunkSize:       800,
		ChunkOverlap:    200,
		NormalizeSpaces: true,
	})
	if err != nil {
		log.Fatalf("ingest: %v", err)
	}

	// ---- Run evaluation ----
	result, err := eval.Evaluate(ctx, pipeline, *evalFile, *topK)
	if err != nil {
		log.Fatalf("evaluation failed: %v", err)
	}

	fmt.Printf("\n=== Evaluation Results ===\n")
	for _, c := range result.CaseResults {
		fmt.Printf("Q: %-45s  P: %.2f  R: %.2f  F1: %.2f\n", c.Query, c.Precision, c.Recall, c.F1Score)
		fmt.Printf("   Matched keywords: %v\n", c.MatchedWords)
	}
	fmt.Printf("\nAverage Precision: %.2f\n", result.AverageP)
	fmt.Printf("Average Recall:    %.2f\n", result.AverageR)
	fmt.Printf("Average F1:        %.2f\n", result.AverageF1)
}
