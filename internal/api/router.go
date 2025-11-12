package api

import (
	"net/http"

	"ragbook/internal/rag"
)

func NewRouter(pipeline *rag.Pipeline) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	mux.Handle("/api/v1/query", queryHandler(pipeline))

	return mux
}
