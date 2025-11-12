package main

import (
	"context"
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
	paramGrid := []struct {
		topK            int
		chunkSize       int
		chunkOverlap    int
		cosineThreshold float32
	}{
		{2, 600, 100, 0.0},
		{3, 800, 200, 0.2},
		{5, 1000, 300, 0.2},
		{3, 800, 200, 0.4},
		{2, 600, 200, 0.3},
	}

	bookPath := "data/book.txt"
	evalPath := "testdata/eval_cases.json"

	text, err := os.ReadFile(bookPath)
	if err != nil {
		log.Fatalf("read book: %v", err)
	}

	bestF1 := 0.0
	var bestParams interface{}

	for _, p := range paramGrid {
		fmt.Printf("\n=== Testing top_k=%d, chunk=%d, overlap=%d, threshold=%.2f ===\n",
			p.topK, p.chunkSize, p.chunkOverlap, p.cosineThreshold)

		embedder := embeddings.NewHashEmbedder(512)
		store := store.NewMemoryStore()
		pipeline := rag.NewPipeline(store, embedder)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		_, err := pipeline.IngestBook(ctx, "alice", string(text), rag.IngestConfig{
			ChunkSize:       p.chunkSize,
			ChunkOverlap:    p.chunkOverlap,
			NormalizeSpaces: true,
		})
		if err != nil {
			log.Fatalf("ingest: %v", err)
		}

		result, err := eval.EvaluateWithThreshold(ctx, pipeline, evalPath, p.topK, p.cosineThreshold)
		if err != nil {
			log.Fatalf("eval: %v", err)
		}

		fmt.Printf("F1: %.3f (Precision %.3f, Recall %.3f)\n",
			result.AverageF1, result.AverageP, result.AverageR)

		if result.AverageF1 > bestF1 {
			bestF1 = result.AverageF1
			bestParams = p
		}
	}

	fmt.Printf("\n Best F1=%.3f with params: %+v\n", bestF1, bestParams)
}
