export interface IndexStats {
  chunks: number;
  vectors: number;
  bm25_docs: number;
  documents: number;
  dim: number;
  embedding_model: string;
  index_version: string;
  status?: string;
}

export interface Calibration {
  in_scope_dense_p10: number;
  in_scope_dense_p50: number;
  out_of_scope_dense_p90: number;
  suggested_min_dense_score: number | null;
}

export interface Aggregates {
  questions: number;
  answerable: number;
  out_of_scope: number;
  errors: number;
  mrr: number;
  answer_rate: number;
  keyword_coverage: number;
  groundedness: number;
  citation_rate: number;
  correct_refusals: number;
  false_refusals: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  stage_ms?: Record<string, number>;
  [key: string]: number | Record<string, number> | undefined;
}

export interface RunSummary {
  id: string;
  created_at: string;
  config: Record<string, string | number>;
  index: IndexStats;
  aggregates: Aggregates;
  calibration: Calibration;
}

export interface Source {
  chunk_id: string;
  /** Present on live answers, omitted in evaluation reports to keep them skimmable. */
  text?: string;
  tokens?: number;
  source: string;
  page: number | null;
  dense: number | null;
  sparse: number | null;
  rank_dense: number | null;
  rank_sparse: number | null;
  fused: number | null;
  rerank: number | null;
}

export interface QuestionResult {
  id: string;
  question: string;
  expect_refusal: boolean;
  refused: boolean;
  refusal_reason: string | null;
  retrieval: Record<string, number>;
  best_dense: number;
  best_sparse: number;
  n_relevant_in_corpus: number;
  answer: string;
  groundedness: number | null;
  keyword_coverage: number;
  judge_score: number | null;
  latency_ms: number;
  error: string | null;
  // Optional on purpose. Reports written before per stage timings and source
  // capture existed do not carry these, and `as RunDetail` on a fetch result
  // is a promise the compiler cannot check: it believed they were always
  // present, so one undefined crashed the whole page to white.
  timings_ms?: Record<string, number>;
  sources?: Source[];
  citations?: number[];
}

export interface RunDetail extends RunSummary {
  results: QuestionResult[];
}

export interface AnswerResponse {
  question: string;
  text: string;
  refused: boolean;
  refusal_reason: string | null;
  citations: number[];
  groundedness: number | null;
  cache_hit: boolean;
  request_id: string;
  timings_ms: Record<string, number>;
  rewritten_question: string | null;
  degraded: boolean;
  degraded_reason: string | null;
  usage: Record<string, number>;
  sources: Source[];
}

export interface DocumentInfo {
  path: string;
  /** True for a visitor's own upload, false for the demo's seed corpus. */
  uploaded?: boolean;
  doc_id: string;
  size_bytes: number;
  chunks: number;
  pages: number;
  tokens: number;
  preview: string;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
  total_chunks: number;
  index_version: string;
  limits: {
    max_upload_mb: number;
    max_documents: number;
    allowed_types: string[];
  };
}

export interface UploadResult {
  file: string;
  size_mb: number;
  status: string;
  chunks: number;
  tokens: number;
  pages: number;
  ingest_ms: number;
  index_version: string;
  total_chunks: number;
  report: Record<string, number>;
  sample_chunks: { chunk_id: string; page: number | null; tokens: number; text: string }[];
}
