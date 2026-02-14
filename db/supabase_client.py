from supabase import create_client, Client
from config import get_settings

_client: Client | None = None


def get_supabase_client() -> Client:
    """Get or create Supabase client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def get_supabase_admin() -> Client:
    """Get Supabase client with service role key for admin operations."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)
