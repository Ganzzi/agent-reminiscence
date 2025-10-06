-- Migration: Add columns for batch update and consolidation refactor
-- Date: 2025-10-04
-- Description:
--   - Add section_id to shortterm_memory_chunk
--   - Add update_count to shortterm_memory
--   - Add last_updated to longterm_memory_chunk

-- ============================================================================
-- SHORTTERM MEMORY CHUNK - Add section_id
-- ============================================================================

-- Add section_id column (nullable)
ALTER TABLE shortterm_memory_chunk
ADD COLUMN IF NOT EXISTS section_id TEXT;

-- Create index for section_id lookups
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_section ON shortterm_memory_chunk (
    shortterm_memory_id,
    section_id
);

-- ============================================================================
-- SHORTTERM MEMORY - Add update_count
-- ============================================================================

-- Add update_count column with default 0
ALTER TABLE shortterm_memory
ADD COLUMN IF NOT EXISTS update_count INTEGER DEFAULT 0;

-- ============================================================================
-- LONGTERM MEMORY CHUNK - Add last_updated
-- ============================================================================

-- Add last_updated column (nullable)
ALTER TABLE longterm_memory_chunk
ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP;

-- Create index for last_updated queries
CREATE INDEX IF NOT EXISTS idx_longterm_chunk_updated ON longterm_memory_chunk (last_updated);

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify columns were added
DO $$
BEGIN
    -- Check shortterm_memory_chunk.section_id
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'shortterm_memory_chunk' 
        AND column_name = 'section_id'
    ) THEN
        RAISE NOTICE 'shortterm_memory_chunk.section_id added successfully';
    ELSE
        RAISE EXCEPTION 'Failed to add shortterm_memory_chunk.section_id';
    END IF;
    
    -- Check shortterm_memory.update_count
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'shortterm_memory' 
        AND column_name = 'update_count'
    ) THEN
        RAISE NOTICE 'shortterm_memory.update_count added successfully';
    ELSE
        RAISE EXCEPTION 'Failed to add shortterm_memory.update_count';
    END IF;
    
    -- Check longterm_memory_chunk.last_updated
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'longterm_memory_chunk' 
        AND column_name = 'last_updated'
    ) THEN
        RAISE NOTICE 'longterm_memory_chunk.last_updated added successfully';
    ELSE
        RAISE EXCEPTION 'Failed to add longterm_memory_chunk.last_updated';
    END IF;
    
    RAISE NOTICE 'Migration completed successfully!';
END $$;