# Fraud Guardian AI — Agent Team Manifest

## Team Lead
- **Role:** Orchestrator, task assignment, progress tracking, decision escalation
- **Responsibilities:** Break MVP into phases, assign tasks, resolve blockers, report to Product Owner

## Architect Teammate
- **Role:** System design, file structure, API contracts, tech stack decisions
- **Scope:** main.py, config.py, api/routes/*, api/middleware/*, overall wiring
- **Key Deliverables:** Wire API stubs to services, implement security middleware, ensure clean request/response contracts

## ML Engineer Teammate
- **Role:** Fraud detection model training, feature engineering, bias detection
- **Scope:** ml/*, models/transaction.py
- **Key Deliverables:** Trained model artifacts, bias report, prediction validation

## Integration Teammate
- **Role:** Third-party API integration, webhook handling, SMS delivery
- **Scope:** integrations/*, api/routes/webhooks.py, services/alert_service.py
- **Key Deliverables:** Paystack webhook processing, Twilio SMS dispatch, end-to-end alert flow

## Compliance & Tester Teammate
- **Role:** Testing, NDPR/CBN compliance, security review, bias auditing
- **Scope:** tests/*, compliance/*, api/middleware/security.py
- **Key Deliverables:** Test suite, compliance checks module, security middleware, bias audit report
