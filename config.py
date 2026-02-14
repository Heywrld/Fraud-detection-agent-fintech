from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    api_key: str = Field(default="dev-api-key", description="API key for dashboard access")

    # Supabase
    supabase_url: str = Field(default="https://placeholder.supabase.co")
    supabase_key: str = Field(default="placeholder-anon-key")
    supabase_service_key: str = Field(default="placeholder-service-key")

    # Paystack
    paystack_secret_key: str = Field(default="sk_test_placeholder")
    paystack_public_key: str = Field(default="pk_test_placeholder")
    paystack_webhook_secret: str = Field(default="whsec_placeholder")

    # Twilio
    twilio_account_sid: str = Field(default="AC_placeholder")
    twilio_auth_token: str = Field(default="placeholder-auth-token")
    twilio_phone_number: str = Field(default="+1234567890")

    # Fraud Detection
    fraud_score_flag_threshold: float = Field(default=0.7, description="Score above which transactions are flagged")
    fraud_score_freeze_threshold: float = Field(default=0.9, description="Score above which accounts are auto-frozen")
    model_path: str = Field(default="ml/artifacts/fraud_model.joblib")

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
