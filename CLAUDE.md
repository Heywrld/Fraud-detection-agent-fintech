# Fraud Guardian AI

Real-time fraud detection agent for Nigerian fintechs.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **ML:** Scikit-learn (IsolationForest), Pandas, NumPy, Joblib
- **Database:** Supabase (PostgreSQL with RLS)
- **Integrations:** Paystack (payments), Twilio (SMS alerts)
- **Config:** Pydantic Settings, python-dotenv
- **Testing:** Pytest + pytest-asyncio

## Project Structure

```
main.py                  # FastAPI app entry point
config.py                # Pydantic Settings (reads .env)
api/routes/              # API endpoints (health, webhooks, transactions, alerts, dashboard)
api/middleware/           # Security middleware (empty)
services/                # Business logic (fraud_engine, transaction, alert, freeze)
ml/                      # ML pipeline (train, predict, features, synthetic_data)
ml/artifacts/            # Trained model files (.joblib) — gitignored
models/                  # Pydantic schemas (transaction, alert, user)
db/                      # Supabase client + schema.sql
integrations/paystack/   # Paystack API client + webhooks
integrations/twilio/     # SMS client + multilingual templates
compliance/              # NDPR/regulatory (not yet implemented)
tests/                   # Tests (not yet written)
```

## Commands

```bash
# Run the server
uvicorn main:app --reload --port 8000

# Train the ML model (must run before first startup)
python -m ml.train

# Run tests
pytest

# Generate synthetic data (included in training)
python -m ml.synthetic_data
```

## Key Architecture Decisions

- **Fraud scoring:** IsolationForest outputs converted to 0-1 probability scale
- **Thresholds:** Flag at 0.7, auto-freeze at 0.9 (configurable in config.py)
- **SMS languages:** English, Hausa, Yoruba, Nigerian Pidgin — routed by state
- **Dev mode:** Twilio client logs SMS instead of sending when `app_env=development`
- **Audit log:** Immutable — enforced by a PL/pgSQL trigger in Supabase
- **Model loading:** Lazy-loaded on app startup; app continues if model is missing

## Current Status

**Implemented:**
- ML pipeline (training, prediction, feature engineering, synthetic data generation)
- Services layer (fraud engine, transactions, alerts, account freeze)
- Database schema with RLS policies
- Paystack + Twilio integrations
- Pydantic models
- Health endpoint

**Phase 3 — Not yet implemented:**
- API route stubs: `webhooks.py`, `transactions.py`, `alerts.py`, `dashboard.py` (return placeholder data)
- Security middleware (`api/middleware/security.py` is empty)
- Compliance module
- Tests

## Environment

Requires a `.env` file (see `.env.example`):
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`
- `PAYSTACK_SECRET_KEY`, `PAYSTACK_PUBLIC_KEY`, `PAYSTACK_WEBHOOK_SECRET`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
- `APP_ENV` (development/staging/production)
- `API_KEY`

## Conventions

- Use `async def` for all route handlers
- Services are synchronous classes with static methods
- All database calls go through `db/supabase_client.py`
- Models use Pydantic v2 (`model_config` style)
- Imports use relative paths from project root
