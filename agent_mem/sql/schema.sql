-- Standalone memory management schema with vector and BM25 search
-- Supports hierarchical memory: Active -> Shortterm -> Longterm
-- ============================================================================
-- EXTENSIONS
-- ============================================================================
-- These must be created by a superuser or user with CREATE privilege
-- Run manually if needed: CREATE EXTENSION IF NOT EXISTS <name> CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS vchord CASCADE;

CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;

CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;

-- BM25 full-text search
-- ============================================================================
-- ACTIVE MEMORY (Working Memory)
-- ============================================================================
-- Active memory represents current working context for an agent.
-- Template-driven structure with sections stored as JSONB.

CREATE TABLE IF NOT EXISTS active_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL, -- Generic agent identifier (UUID, string, int)
    title VARCHAR(500) NOT NULL,
    template_content TEXT NOT NULL, -- YAML template defining structure
    sections JSONB DEFAULT '{}', -- {section_id: {content: str, update_count: int}}
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for querying by external_id (primary access pattern)
CREATE INDEX IF NOT EXISTS idx_active_memory_external_id ON active_memory (external_id);

-- Index for JSONB sections
CREATE INDEX IF NOT EXISTS idx_active_memory_sections ON active_memory USING gin (sections);

-- JSONB index for metadata queries
CREATE INDEX IF NOT EXISTS idx_active_memory_metadata ON active_memory USING gin (metadata);

-- ============================================================================
-- SHORTTERM MEMORY (Recent Searchable Knowledge)
-- ============================================================================
-- Shortterm memory summary table - represents consolidated active memories.

CREATE TABLE IF NOT EXISTS shortterm_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    update_count INTEGER DEFAULT 0, -- Track number of updates for promotion threshold
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for external_id queries
CREATE INDEX IF NOT EXISTS idx_shortterm_memory_external_id ON shortterm_memory (external_id);

-- JSONB index for metadata queries
CREATE INDEX IF NOT EXISTS idx_shortterm_memory_metadata ON shortterm_memory USING gin (metadata);

-- Time-based index for temporal queries
CREATE INDEX IF NOT EXISTS idx_shortterm_memory_dates ON shortterm_memory (
    external_id,
    created_at,
    last_updated
);

-- ============================================================================
-- SHORTTERM MEMORY CHUNKS (Searchable Chunks with Vectors)
-- ============================================================================
-- Chunked content with vector embeddings and BM25 for hybrid search.

CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_memory_id INTEGER NOT NULL REFERENCES shortterm_memory (id) ON DELETE CASCADE,
    external_id VARCHAR(255) NOT NULL,
    chunk_order INTEGER NOT NULL, -- Order of chunk within memory
    content TEXT NOT NULL,
    embedding vector (768), -- Vector dimension configurable (default 768 for nomic-embed-text)
    content_bm25 bm25vector, -- Auto-populated by trigger
    section_id TEXT, -- References active memory section (nullable, for tracking source)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for foreign key lookups
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_memory_id ON shortterm_memory_chunk (shortterm_memory_id);

-- Index for external_id queries
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_external_id ON shortterm_memory_chunk (external_id);

-- Index for ordering chunks
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_order ON shortterm_memory_chunk (
    shortterm_memory_id,
    chunk_order
);

-- Vector similarity search index using HNSW (Hierarchical Navigable Small World)
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_vector ON shortterm_memory_chunk USING hnsw (embedding vector_cosine_ops);

-- BM25 keyword search index
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_bm25 ON shortterm_memory_chunk USING bm25 (content_bm25 bm25_ops);

-- JSONB index for metadata
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_metadata ON shortterm_memory_chunk USING gin (metadata);

-- Index for section_id lookups
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_section ON shortterm_memory_chunk (
    shortterm_memory_id,
    section_id
);

-- ============================================================================
-- LONGTERM MEMORY CHUNKS (Consolidated Knowledge with Temporal Tracking)
-- ============================================================================
-- Promoted shortterm memories with temporal validity tracking.

CREATE TABLE IF NOT EXISTS longterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    shortterm_memory_id INTEGER, -- Reference to source (optional)
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector (768),
    content_bm25 bm25vector,
    confidence_score REAL DEFAULT 0.5, -- Confidence in information accuracy
    importance_score REAL DEFAULT 0.5, -- Importance for prioritization
    metadata JSONB DEFAULT '{}',
    start_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, -- When information became valid
    end_date TIMESTAMP, -- NULL means still valid, set to mark as superseded
    last_updated TIMESTAMP, -- Track when chunk was last updated from shortterm
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for external_id queries
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_external_id ON longterm_memory_chunk (external_id);

-- Index for finding currently valid chunks
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_valid ON longterm_memory_chunk (
    external_id,
    start_date,
    end_date
)
WHERE
    end_date IS NULL;

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_temporal ON longterm_memory_chunk (
    external_id,
    start_date DESC,
    end_date DESC
);

-- Index for importance-based retrieval
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_importance ON longterm_memory_chunk (
    external_id,
    importance_score DESC,
    confidence_score DESC
);

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_vector ON longterm_memory_chunk USING hnsw (embedding vector_cosine_ops);

-- BM25 keyword search index
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_bm25 ON longterm_memory_chunk USING bm25 (content_bm25 bm25_ops);

-- JSONB index for metadata
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_metadata ON longterm_memory_chunk USING gin (metadata);

-- Index for last_updated queries
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_updated ON longterm_memory_chunk (last_updated);

-- ============================================================================
-- TRIGGERS
-- ============================================================================
-- Create a BERT tokenizer for BM25 indexing
SELECT create_tokenizer (
        'bert', $$ model = "bert_base_uncased" $$
    );

-- Function to automatically update BM25 vectors when content changes
CREATE
OR
REPLACE
    FUNCTION update_bm25_vector_chunk () RETURNS TRIGGER AS $$ BEGIN IF TG_OP = 'INSERT'
    OR (
        TG_OP = 'UPDATE'
        AND OLD.content IS DISTINCT
        FROM NEW.content
    ) THEN
    -- Tokenize content using BERT tokenizer for BM25 indexing
    NEW.content_bm25 = tokenize (NEW.content, 'bert');

END IF;

RETURN NEW;

END;

$$ LANGUAGE plpgsql;

-- Trigger for shortterm_memory_chunk
DROP TRIGGER IF EXISTS trigger_update_bm25_shortterm_chunk ON shortterm_memory_chunk;

CREATE TRIGGER trigger_update_bm25_shortterm_chunk
    BEFORE INSERT OR UPDATE ON shortterm_memory_chunk
    FOR EACH ROW 
    EXECUTE FUNCTION update_bm25_vector_chunk();

-- Trigger for longterm_memory_chunk
DROP TRIGGER IF EXISTS trigger_update_bm25_longterm_chunk ON longterm_memory_chunk;

CREATE TRIGGER trigger_update_bm25_longterm_chunk
    BEFORE INSERT OR UPDATE ON longterm_memory_chunk
    FOR EACH ROW 
    EXECUTE FUNCTION update_bm25_vector_chunk();

-- Function to update updated_at timestamp
CREATE
OR
REPLACE
    FUNCTION update_updated_at_column () RETURNS TRIGGER AS $$ BEGIN
    -- Handle both updated_at and last_updated column names
    IF TG_TABLE_NAME = 'shortterm_memory' THEN NEW.last_updated = CURRENT_TIMESTAMP;

ELSE NEW.updated_at = CURRENT_TIMESTAMP;

END IF;

RETURN NEW;

END;

$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at for active_memory
DROP TRIGGER IF EXISTS trigger_update_active_memory_timestamp ON active_memory;

CREATE TRIGGER trigger_update_active_memory_timestamp
    BEFORE UPDATE ON active_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to auto-update last_updated for shortterm_memory
DROP TRIGGER IF EXISTS trigger_update_shortterm_memory_timestamp ON shortterm_memory;

CREATE TRIGGER trigger_update_shortterm_memory_timestamp
    BEFORE UPDATE ON shortterm_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- NOTE: update_count is now tracked per section in the sections JSONB field
-- Each section in the sections JSONB has its own update_count:
-- sections = {
--   "section_id": {
--     "content": "...",
--     "update_count": 5
--   }
-- }
-- ============================================================================
-- HELPER VIEWS
-- ============================================================================
-- View to see currently valid longterm chunks
CREATE OR REPLACE VIEW valid_longterm_chunks AS
SELECT
    id,
    external_id,
    content,
    confidence_score,
    importance_score,
    start_date,
    created_at
FROM longterm_memory_chunk
WHERE
    end_date IS NULL
ORDER BY
    importance_score DESC,
    confidence_score DESC;

-- ============================================================================
-- NOTES
-- ============================================================================
-- 1. Vector dimension (768) matches nomic-embed-text default
--    Adjust if using a different embedding model
--
-- 2. HNSW index parameters can be tuned for performance:
--    - m: max connections per layer (default 16)
--    - ef_construction: size of dynamic candidate list (default 64)
--    Example: USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
--
-- 3. BM25 uses BERT tokenizer by default
--    Can create additional tokenizers for different languages
--
-- 4. Active memory uses template-driven structure:
--    - template_content: YAML template defining section IDs and structure
--    - sections: JSONB with {section_id: {content: str, update_count: int}}
--    - Each section tracks its own update_count for consolidation triggers
--
-- 5. To enable extensions, run as superuser:
--    CREATE EXTENSION IF NOT EXISTS vector;

-- CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;

-- CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;