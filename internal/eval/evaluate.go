package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"ragbook/internal/rag"
	"ragbook/internal/types"
)

// TestCase defines one evaluation query.
type TestCase struct {
	Query            string   `json:"query"`
	ExpectedKeywords []string `json:"expected_keywords"`
}

// Result stores evaluation metrics.
type Result struct {
	TotalCases  int
	AverageP    float64
	AverageR    float64
	AverageF1   float64
	CaseResults []CaseResult
}

// CaseResult stores metrics for one test case.
type CaseResult struct {
	Query          string
	Precision      float64
	Recall         float64
	F1Score        float64
	MatchedCount   int
	ExpectedCount  int
	RetrievedCount int
	MatchedWords   []string
}

// Evaluate runs all test cases and computes precision/recall/F1.
func Evaluate(ctx context.Context, pipeline *rag.Pipeline, jsonPath string, topK int) (*Result, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read eval cases: %w", err)
	}

	var cases []TestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse eval cases: %w", err)
	}

	results := Result{TotalCases: len(cases)}

	for _, tc := range cases {
		req := types.QueryRequest{Query: tc.Query, TopK: topK}
		resp, err := pipeline.AnswerQuery(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("query failed: %w", err)
		}

		retrievedText := strings.ToLower(strings.Join(extractTexts(resp.Sources), " "))
		matchedWords := make(map[string]bool)

		for _, kw := range tc.ExpectedKeywords {
			if strings.Contains(retrievedText, strings.ToLower(kw)) {
				matchedWords[kw] = true
			}
		}

		matched := len(matchedWords)
		expected := len(tc.ExpectedKeywords)
		retrieved := len(resp.Sources)

		// Avoid division by zero
		if retrieved == 0 {
			retrieved = 1
		}
		if expected == 0 {
			expected = 1
		}

		precision := float64(matched) / float64(retrieved)
		if precision > 1.0 {
			precision = 1.0
		}

		recall := float64(matched) / float64(len(tc.ExpectedKeywords))
		if recall > 1.0 {
			recall = 1.0
		}

		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * (precision * recall) / (precision + recall)
		}

		results.CaseResults = append(results.CaseResults, CaseResult{
			Query:          tc.Query,
			Precision:      precision,
			Recall:         recall,
			F1Score:        f1,
			MatchedCount:   matched,
			ExpectedCount:  len(tc.ExpectedKeywords),
			RetrievedCount: len(resp.Sources),
			MatchedWords:   mapKeys(matchedWords),
		})

		results.AverageP += precision
		results.AverageR += recall
		results.AverageF1 += f1
	}

	n := float64(results.TotalCases)
	if n > 0 {
		results.AverageP /= n
		results.AverageR /= n
		results.AverageF1 /= n
	}

	return &results, nil
}

func extractTexts(srcs []types.SourceChunk) []string {
	texts := make([]string, len(srcs))
	for i, s := range srcs {
		texts[i] = s.Text
	}
	return texts
}

func mapKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
