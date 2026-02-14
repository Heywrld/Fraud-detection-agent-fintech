"""
Multilingual SMS templates for fraud alert notifications.

Supports English (en), Hausa (ha), Yoruba (yo), and Nigerian Pidgin (pcm).
Templates are selected based on the customer's Nigerian state of residence
to maximise comprehension.
"""

import logging
from typing import Optional

logger = logging.getLogger("fraud_guardian.twilio.templates")

# ---------------------------------------------------------------------------
# Template definitions
# Keys: (template_type, language_code)
# Values: template strings with {placeholders} for str.format()
# ---------------------------------------------------------------------------

TEMPLATES: dict[tuple[str, str], str] = {
    # ---- fraud_alert ----
    ("fraud_alert", "en"): (
        "Suspicious activity detected on your account. "
        "N{amount} {channel} transaction. "
        "Reply FREEZE to lock your account."
    ),
    ("fraud_alert", "ha"): (
        "An gano wani aiki mai ban tsoro a asusunka. "
        "Ma'amala ta {channel} ta N{amount}. "
        "Ka amsa FREEZE don kulle asusunka."
    ),
    ("fraud_alert", "yo"): (
        "A ti ri iwa ifura lori akanti re. "
        "Idunadura {channel} ti N{amount}. "
        "Fi FREEZE ranse lati di akanti re."
    ),
    ("fraud_alert", "pcm"): (
        "Dem don see suspicious movement for your account. "
        "N{amount} {channel} transaction. "
        "Reply FREEZE to lock your account."
    ),

    # ---- account_frozen ----
    ("account_frozen", "en"): (
        "Your account has been frozen due to suspicious activity. "
        "Contact support to verify your identity."
    ),
    ("account_frozen", "ha"): (
        "An daskarar da asusunka saboda wani aiki mai ban tsoro. "
        "Ka tuntubar ma'aikatan tallafi don tabbatar da asalinka."
    ),
    ("account_frozen", "yo"): (
        "A ti di akanti re nitori iwa ifura. "
        "Kan si apa atileyin lati jeri idanimo re."
    ),
    ("account_frozen", "pcm"): (
        "Dem don freeze your account because of suspicious activity. "
        "Contact support to verify say na you."
    ),

    # ---- transaction_flagged ----
    ("transaction_flagged", "en"): (
        "A transaction of N{amount} was flagged for review. "
        "If this wasn't you, reply FREEZE."
    ),
    ("transaction_flagged", "ha"): (
        "An yi wa ma'amalar N{amount} alama don dubawa. "
        "Idan ba kai ba ne, ka amsa FREEZE."
    ),
    ("transaction_flagged", "yo"): (
        "Idunadura N{amount} ti a fi ami si fun atunyewo. "
        "Ti kii se iwo, fi FREEZE ranse."
    ),
    ("transaction_flagged", "pcm"): (
        "One transaction of N{amount} don dey flagged for review. "
        "If no be you do am, reply FREEZE."
    ),
}

# ---------------------------------------------------------------------------
# Nigerian state -> language mapping
# ---------------------------------------------------------------------------

# Northern states where Hausa is the dominant language
HAUSA_STATES = {
    "kano", "kaduna", "sokoto", "katsina", "jigawa",
    "zamfara", "kebbi", "bauchi", "gombe", "yobe",
    "borno", "adamawa",
}

# South-western states where Yoruba is dominant, plus Kwara and Kogi
# which have significant Yoruba-speaking populations
YORUBA_STATES = {
    "lagos", "oyo", "osun", "ondo", "ekiti", "ogun",
    "kwara", "kogi",
}


def get_language_for_state(state: Optional[str]) -> str:
    """
    Determine the most likely language for a Nigerian state.

    Mapping:
        - Kano, Kaduna, Sokoto, Katsina, Jigawa, Zamfara, Kebbi,
          Bauchi, Gombe, Yobe, Borno, Adamawa -> Hausa ("ha")
        - Lagos, Oyo, Osun, Ondo, Ekiti, Ogun, Kwara, Kogi -> Yoruba ("yo")
        - All other states / None -> English ("en")

    Args:
        state: Nigerian state name (case-insensitive) or None.

    Returns:
        ISO 639 language code string: "ha", "yo", or "en".
    """
    if not state:
        return "en"

    normalised = state.strip().lower()

    if normalised in HAUSA_STATES:
        return "ha"
    if normalised in YORUBA_STATES:
        return "yo"

    return "en"


def get_template(template_type: str, language: str, **kwargs) -> str:
    """
    Render an SMS template for the given type and language.

    If the requested template+language combination does not exist,
    falls back to English. If even the English template is missing,
    returns a generic fallback message.

    Args:
        template_type: One of 'fraud_alert', 'account_frozen', 'transaction_flagged'.
        language: ISO 639 language code ('en', 'ha', 'yo', 'pcm').
        **kwargs: Values to substitute into the template (e.g. amount, channel).

    Returns:
        Rendered SMS message string.
    """
    # Try the requested language first
    template = TEMPLATES.get((template_type, language))

    # Fallback to English if the language is not available
    if template is None:
        logger.warning(
            "Template '%s' not found for language '%s', falling back to English",
            template_type,
            language,
        )
        template = TEMPLATES.get((template_type, "en"))

    # Final fallback if template type is completely unknown
    if template is None:
        logger.error("Unknown template type: '%s'", template_type)
        return (
            "Alert: Suspicious activity detected on your account. "
            "Please contact support immediately."
        )

    try:
        return template.format(**kwargs)
    except KeyError as exc:
        logger.error(
            "Missing template variable %s for template '%s' (%s). kwargs=%s",
            exc,
            template_type,
            language,
            kwargs,
        )
        # Return the template with unfilled placeholders rather than crashing
        return template
