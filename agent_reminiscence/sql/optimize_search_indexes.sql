-- Search Performance Optimization Indexes
-- Issue: PREAAAA-20 - Search performance profiling + optimization
-- Created: 2026-04-30

-- ============================================================================
-- PROFILING FINDINGS
-- ============================================================================
-- Current indexes exist but are suboptimal for hybrid search queries:
-- 1. idx_shortterm_chunk_external_id - single column, not composite
-- 2. idx_longterm_chunk_external_id - single column, not composite
-- 3. HNSW indexes exist but not optimized for filtered queries
-- 4. BM25 indexes exist but not composite with external_id
--
-- BOTTLENECK IDENTIFIED:
-- The hybrid_search query filters by external_id THEN does vector/BM25 search.
-- Current separate indexes require full scan per external_id, then vector scan.
-- ============================================================================

-- ============================================================================
-- SHORTTERM MEMORY CHUNK OPTIMIZATIONS
-- ============================================================================

-- Composite index for vector search with external_id filter
-- Supports: WHERE external_id = X AND embedding IS NOT NULL
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_external_id_vector
ON shortterm_memory_chunk (external_id, embedding)
USING hnsw (embedding vector_cosine_ops)
WITH (m = 8, ef_construction = 64);

-- Composite index for BM25 search with external_id filter
-- Supports: WHERE external_id = X AND content_bm25 IS NOT NULL
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_external_id_bm25
ON shortterm_memory_chunk (external_id, content_bm25)
USING bm25 (content_bm25 bm25_ops);

-- Covering index for common retrieval pattern
-- Includes frequently accessed columns to avoid table lookups
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_covering
ON shortterm_memory_chunk (external_id, shortterm_memory_id)
INCLUDE (content, section_id, metadata, access_count);

-- ============================================================================
-- LONGTERM MEMORY CHUNK OPTIMIZATIONS
-- ============================================================================

-- Composite index for vector search with external_id filter
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_external_id_vector
ON longterm_memory_chunk (external_id, embedding)
USING hnsw (embedding vector_cosine_ops)
WITH (m = 8, ef_construction = 64);

-- Composite index for BM25 search with external_id filter
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_external_id_bm25
ON longterm_memory_chunk (external_id, content_bm25)
USING bm25 (content_bm25 bm25_ops);

-- Covering index for common retrieval pattern
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_covering
ON longterm_memory_chunk (external_id, importance)
INCLUDE (content, start_date, last_updated, metadata);

-- ============================================================================
-- ANALYZE TABLES FOR QUERY PLANNER
-- ============================================================================
ANALYZE shortterm_memory_chunk;
ANALYZE longterm_memory_chunk;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Check index usage:
-- EXPLAIN ANALYZE SELECT ... FROM shortterm_memory_chunk
--   WHERE external_id = 'test-agent' AND embedding IS NOT NULL;

-- Check index sizes:
-- SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
-- FROM pg_indexes WHERE tablename IN ('shortterm_memory_chunk', 'longterm_memory_chunk');