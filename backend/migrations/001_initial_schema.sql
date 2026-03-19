-- backend/migrations/001_initial_schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    strategy_selected VARCHAR(50) NOT NULL,
    hallucination_risk FLOAT NOT NULL,
    confidence_score FLOAT NOT NULL,
    trust_score FLOAT NOT NULL,
    tokens_used INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    final_answer TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reasoning_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES requests(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    entropy FLOAT,
    flagged BOOLEAN DEFAULT FALSE,
    assumptions JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX idx_requests_created_at ON requests(created_at);
CREATE INDEX idx_reasoning_steps_request_id ON reasoning_steps(request_id);