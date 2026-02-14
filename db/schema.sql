-- Fraud Guardian AI — Supabase Schema
-- Run this in the Supabase SQL Editor to create all tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- TRANSACTIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paystack_ref TEXT UNIQUE NOT NULL,
    amount_ngn NUMERIC(15, 2) NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('pos', 'mobile_money', 'ussd', 'bank_transfer', 'card')),
    customer_id TEXT NOT NULL,
    customer_phone_hash TEXT,
    location_state TEXT,
    location_lga TEXT,
    device_fingerprint TEXT,
    fraud_score FLOAT DEFAULT 0.0,
    is_flagged BOOLEAN DEFAULT FALSE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'flagged', 'frozen')),
    -- CBN Compliance fields
    cbn_risk_level TEXT CHECK (cbn_risk_level IN ('low', 'medium', 'high', 'critical')),
    cbn_risk_score INTEGER,
    cbn_red_flags JSONB DEFAULT '[]',
    file_str BOOLEAN DEFAULT FALSE,
    cbn_recommendation TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_is_flagged ON transactions(is_flagged);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_score ON transactions(fraud_score);
CREATE INDEX IF NOT EXISTS idx_transactions_cbn_risk_level ON transactions(cbn_risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_file_str ON transactions(file_str);

-- ============================================
-- ALERTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    alert_type TEXT NOT NULL CHECK (alert_type IN ('sms', 'freeze', 'manual_review')),
    language TEXT DEFAULT 'en' CHECK (language IN ('en', 'ha', 'yo', 'pcm')),
    message TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed', 'resolved')),
    sent_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

-- ============================================
-- AUDIT LOG TABLE (Immutable)
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id UUID,
    details JSONB DEFAULT '{}',
    ip_address TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Prevent updates and deletes on audit_log (immutability)
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log records cannot be modified or deleted';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_immutable_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();

CREATE TRIGGER audit_log_immutable_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();

CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at DESC);

-- ============================================
-- MODEL METRICS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version TEXT NOT NULL,
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1 FLOAT,
    bias_report JSONB DEFAULT '{}',
    trained_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ROW LEVEL SECURITY (NDPR Compliance)
-- ============================================
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;

-- Service role can do everything
CREATE POLICY "Service role full access on transactions" ON transactions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on alerts" ON alerts
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on audit_log" ON audit_log
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on model_metrics" ON model_metrics
    FOR ALL USING (auth.role() = 'service_role');

-- Anon key can only read (for dashboard)
CREATE POLICY "Anon read transactions" ON transactions
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Anon read alerts" ON alerts
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Anon read audit_log" ON audit_log
    FOR SELECT USING (auth.role() = 'anon');
