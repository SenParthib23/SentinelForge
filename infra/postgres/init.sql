-- ============================================================
-- SentinelForge — PostgreSQL Schema Initialization
-- ============================================================

-- 1. Transactions (raw event log)
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(64) UNIQUE NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'INR',
    channel VARCHAR(20) NOT NULL,           -- 'card', 'upi', 'neft', 'imps'
    device_fingerprint VARCHAR(128),
    ip_address VARCHAR(45),
    location_lat DECIMAL(10, 7),
    location_lng DECIMAL(10, 7),
    merchant_category VARCHAR(64),
    is_international BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Feature snapshots (offline feature store)
CREATE TABLE IF NOT EXISTS feature_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id),
    features JSONB NOT NULL,                -- full feature vector as JSON
    feature_version VARCHAR(16) NOT NULL,   -- feature definition version
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Fraud scores (model predictions)
CREATE TABLE IF NOT EXISTS fraud_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id),
    model_version VARCHAR(64) NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_label VARCHAR(20) NOT NULL,        -- 'low', 'medium', 'high', 'critical'
    threshold_used FLOAT NOT NULL,
    was_flagged BOOLEAN NOT NULL,
    scoring_latency_ms FLOAT NOT NULL,
    feature_fetch_latency_ms FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Fraud cases (flagged for investigation)
CREATE TABLE IF NOT EXISTS fraud_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id),
    risk_score FLOAT NOT NULL,
    case_context JSONB,                     -- assembled context bundle
    investigation_narrative TEXT,           -- LLM-generated narrative
    narrative_model_version VARCHAR(64),
    analyst_decision VARCHAR(20),           -- 'approved', 'rejected', 'escalated'
    analyst_notes TEXT,
    analyst_id VARCHAR(64),
    decision_timestamp TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending',   -- 'pending', 'investigating', 'resolved'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Model versions (registry metadata)
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(64) NOT NULL,        -- 'lightgbm_fraud' or 'investigation_llm'
    version VARCHAR(32) NOT NULL,
    mlflow_run_id VARCHAR(64),
    base_model VARCHAR(128),
    dataset_hash VARCHAR(64),
    hyperparameters JSONB,
    eval_metrics JSONB,                     -- benchmark + regression results
    status VARCHAR(20) DEFAULT 'candidate', -- 'candidate', 'canary', 'production', 'retired'
    canary_start TIMESTAMP WITH TIME ZONE,
    canary_end TIMESTAMP WITH TIME ZONE,
    canary_metrics JSONB,
    promoted_at TIMESTAMP WITH TIME ZONE,
    retired_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Labeled dataset (analyst feedback for retraining)
CREATE TABLE IF NOT EXISTS labeled_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(64) NOT NULL,
    features JSONB NOT NULL,
    risk_score FLOAT NOT NULL,
    analyst_label VARCHAR(20) NOT NULL,     -- 'fraud', 'legitimate', 'suspicious'
    label_source VARCHAR(20) NOT NULL,      -- 'analyst', 'rule', 'confirmed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Drift snapshots (for monitoring)
CREATE TABLE IF NOT EXISTS drift_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(64) NOT NULL,
    snapshot_type VARCHAR(20) NOT NULL,     -- 'feature_drift', 'prediction_drift'
    drift_detected BOOLEAN NOT NULL,
    drift_score FLOAT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for high-performance lookup
CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_txn_device ON transactions(device_fingerprint, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scores_txn ON fraud_scores(transaction_id);
CREATE INDEX IF NOT EXISTS idx_cases_status ON fraud_cases(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_labeled_created ON labeled_data(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_status ON model_versions(model_name, status);
