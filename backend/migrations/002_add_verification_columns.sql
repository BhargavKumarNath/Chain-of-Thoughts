-- 002_add_verification_columns.sql
-- Adds verification pipeline columns. Backward-compatible: existing rows get NULLs.

ALTER TABLE requests
    ADD COLUMN IF NOT EXISTS verification_status VARCHAR(10) DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS verification_confidence FLOAT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS difficulty_level VARCHAR(10) DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS retry_used BOOLEAN DEFAULT FALSE;

-- Index for filtering by verification status
CREATE INDEX IF NOT EXISTS idx_requests_verification_status ON requests(verification_status);
CREATE INDEX IF NOT EXISTS idx_requests_difficulty_level ON requests(difficulty_level);
