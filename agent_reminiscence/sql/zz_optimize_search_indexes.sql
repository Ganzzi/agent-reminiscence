-- Search Performance Optimization Indexes
-- Issue: PREAAAA-20 - Search performance profiling + optimization
-- Created: 2026-04-30
-- Repaired under NGU-98 (QA-HARDENING-005-02):
--   * Prefixed zz_ because the PostgreSQL entrypoint executes
--     /docker-entrypoint-initdb.d files in alphabetical order; this file
--     previously sorted before schema.sql and referenced tables that do not
--     exist yet at that point.
--   * The four composite ANN statements (hnsw/bm25 keyed on a leading
--     external_id column) were withdrawn: their grammar was invalid
--     (USING must precede the column list) and neither access method
--     supports composite keys. Supported single-column equivalents already
--     exist in schema.sql (idx_*_chunk_vector, idx_*_chunk_bm25); duplicate
--     indexes are intentionally not created.

-- ============================================================================
-- SHORTTERM MEMORY CHUNK OPTIMIZATIONS
-- ============================================================================

-- Covering index for common retrieval pattern
-- Includes frequently accessed columns to avoid table lookups
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_covering
ON shortterm_memory_chunk (external_id, shortterm_memory_id)
INCLUDE (content, section_id, metadata, access_count);

-- ============================================================================
-- LONGTERM MEMORY CHUNK OPTIMIZATIONS
-- ============================================================================

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
