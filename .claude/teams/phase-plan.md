# MVP Phase Plan

## Phase 1-2: COMPLETE (already built)
- ML pipeline (train, predict, features, synthetic_data)
- Services layer (fraud_engine, transaction, alert, freeze)
- Database schema (Supabase + RLS)
- Integrations scaffolding (Paystack client, Twilio client + templates)
- Pydantic models
- FastAPI skeleton + health endpoint

## Phase 3A: Model Training + Webhook Wiring (parallel)
**ML Engineer:**
- [ ] Train IsolationForest model, generate artifacts in ml/artifacts/
- [ ] Validate predictions on synthetic data, check bias across states/channels

**Integration Teammate:**
- [ ] Implement Paystack webhook handler (signature verification, event routing)
- [ ] Wire webhook events to fraud_engine for real-time scoring

## Phase 3B: API Endpoints + Security (parallel, after 3A)
**Architect:**
- [ ] Wire transactions.py routes to transaction_service
- [ ] Wire alerts.py routes to alert_service
- [ ] Wire dashboard.py routes with aggregation queries

**Compliance & Tester:**
- [ ] Implement security middleware (API key auth, webhook signature verification)
- [ ] Begin NDPR compliance module

## Phase 3C: Testing + Final Review (after 3B)
**Compliance & Tester:**
- [ ] Unit tests for services layer
- [ ] Integration tests for API endpoints
- [ ] Security review (OWASP top 10 check)
- [ ] Bias audit on trained model

**Team Lead:**
- [ ] Final review, summary to Product Owner
