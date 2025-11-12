package types

// DocumentChunk represents a chunk of the book with its embedding.
type DocumentChunk struct {
	ID        string    `json:"id"`
	BookID    string    `json:"book_id"`
	Index     int       `json:"index"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"-"` // not serialized
}

// QueryRequest is the JSON payload for /api/v1/query.
type QueryRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

// SourceChunk represents a retrieved chunk with similarity score.
type SourceChunk struct {
	ID     string  `json:"id"`
	BookID string  `json:"book_id"`
	Index  int     `json:"index"`
	Score  float32 `json:"score"`
	Text   string  `json:"text"`
}

// QueryResponse is returned by /api/v1/query.
type QueryResponse struct {
	Answer  string        `json:"answer"`
	Sources []SourceChunk `json:"sources"`
}
