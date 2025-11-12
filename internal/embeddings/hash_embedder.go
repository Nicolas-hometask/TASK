package embeddings

import (
	"hash/fnv"
	"strings"
	"unicode"
)

// Embedder is the interface used by the RAG pipeline.
type Embedder interface {
	Embed(texts []string) ([][]float32, error)
	EmbedQuery(text string) ([]float32, error)
}

// HashEmbedder is a minimal, self-contained embedding model.
// It hashes tokens into a fixed-size vector.
type HashEmbedder struct {
	dim int
}

// NewHashEmbedder creates a new hash-based embedder.
func NewHashEmbedder(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 512
	}
	return &HashEmbedder{dim: dim}
}

// Embed multiple texts.
func (h *HashEmbedder) Embed(texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		result[i] = h.embedSingle(t)
	}
	return result, nil
}

// EmbedQuery embeds a single query.
func (h *HashEmbedder) EmbedQuery(text string) ([]float32, error) {
	return h.embedSingle(text), nil
}

func (h *HashEmbedder) embedSingle(text string) []float32 {
	vec := make([]float32, h.dim)
	tokens := simpleTokenize(text)
	if len(tokens) == 0 {
		return vec
	}
	for _, tok := range tokens {
		idx := h.hashToken(tok)
		vec[idx] += 1.0
	}
	normalize(vec)
	return vec
}

func (h *HashEmbedder) hashToken(tok string) int {
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(tok))
	return int(hasher.Sum32() % uint32(h.dim))
}

func simpleTokenize(text string) []string {
	var b strings.Builder
	var tokens []string
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			if b.Len() > 0 {
				tokens = append(tokens, b.String())
				b.Reset()
			}
		}
	}
	if b.Len() > 0 {
		tokens = append(tokens, b.String())
	}
	return tokens
}

func normalize(vec []float32) {
	var sumSquares float32
	for _, v := range vec {
		sumSquares += v * v
	}
	if sumSquares == 0 {
		return
	}
	inv := 1 / float32Sqrt(sumSquares)
	for i, v := range vec {
		vec[i] = v * inv
	}
}

func float32Sqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
