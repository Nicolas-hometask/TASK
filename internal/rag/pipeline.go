package rag

import (
	"context"
	"fmt"
	"strings"

	"ragbook/internal/embeddings"
	"ragbook/internal/store"
	"ragbook/internal/types"
)

type IngestConfig struct {
	ChunkSize       int
	ChunkOverlap    int
	MaxChunks       int
	NormalizeSpaces bool
}

type Pipeline struct {
	store    store.VectorStore
	embedder embeddings.Embedder
}

func NewPipeline(store store.VectorStore, embedder embeddings.Embedder) *Pipeline {
	return &Pipeline{store: store, embedder: embedder}
}

func (p *Pipeline) IngestBook(ctx context.Context, bookID, text string, cfg IngestConfig) (int, error) {
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = 800
	}
	if cfg.ChunkOverlap < 0 {
		cfg.ChunkOverlap = 0
	}

	if cfg.NormalizeSpaces {
		text = normalizeSpaces(text)
	}

	chunks := chunkText(text, cfg.ChunkSize, cfg.ChunkOverlap)
	if cfg.MaxChunks > 0 && len(chunks) > cfg.MaxChunks {
		chunks = chunks[:cfg.MaxChunks]
	}

	embs, err := p.embedder.Embed(chunks)
	if err != nil {
		return 0, fmt.Errorf("embedding chunks: %w", err)
	}

	for i, chunkText := range chunks {
		id := fmt.Sprintf("%s-%d", bookID, i)
		chunk := types.DocumentChunk{
			ID:        id,
			BookID:    bookID,
			Index:     i,
			Text:      chunkText,
			Embedding: embs[i],
		}
		if err := p.store.AddChunk(chunk); err != nil {
			return i, fmt.Errorf("adding chunk %d: %w", i, err)
		}
	}
	return len(chunks), nil
}

func (p *Pipeline) AnswerQuery(ctx context.Context, req types.QueryRequest) (*types.QueryResponse, error) {
	topK := req.TopK
	if topK <= 0 {
		topK = 5
	}

	emb, err := p.embedder.EmbedQuery(req.Query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	sources, err := p.store.Search(emb, topK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	answer := buildSimpleAnswer(req.Query, sources)

	resp := &types.QueryResponse{
		Answer:  answer,
		Sources: sources,
	}
	return resp, nil
}

func buildSimpleAnswer(query string, sources []types.SourceChunk) string {
	if len(sources) == 0 {
		return "No relevant passages found for your query."
	}
	var b strings.Builder
	b.WriteString("Answer based on the book:\n\n")
	b.WriteString("Query: " + query + "\n\n")
	for i, s := range sources {
		fmt.Fprintf(&b, "Excerpt %d (chunk %d, score=%.3f):\n%s\n\n", i+1, s.Index, s.Score, s.Text)
	}
	b.WriteString("Note: These excerpts were retrieved from the source text.\n")
	return b.String()
}

func chunkText(text string, size, overlap int) []string {
	if size <= 0 {
		return []string{text}
	}
	if overlap < 0 {
		overlap = 0
	}
	var chunks []string
	runes := []rune(text)
	n := len(runes)
	for start := 0; start < n; {
		end := start + size
		if end > n {
			end = n
		}
		chunks = append(chunks, string(runes[start:end]))
		if end == n {
			break
		}
		start = end - overlap
		if start < 0 {
			start = 0
		}
	}
	return chunks
}

func normalizeSpaces(s string) string {
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}
