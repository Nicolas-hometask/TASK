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

// EvaluateWithThreshold runs the normal evaluation but ignores results below cosine threshold.
func EvaluateWithThreshold(
	ctx context.Context,
	pipeline *rag.Pipeline,
	jsonPath string,
	topK int,
	cosineThreshold float32,
) (*Result, error) {

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

		// filter retrieved chunks
		filtered := make([]types.SourceChunk, 0)
		for _, s := range resp.Sources {
			if s.Score >= cosineThreshold {
				filtered = append(filtered, s)
			}
		}
		resp.Sources = filtered

		// Evaluate relevance
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
		if retrieved == 0 {
			retrieved = 1
		}
		if expected == 0 {
			expected = 1
		}

		precision := float64(matched) / float64(retrieved)
		recall := float64(matched) / float64(len(tc.ExpectedKeywords))
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
