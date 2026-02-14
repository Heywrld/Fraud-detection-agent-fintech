"""
Twilio SMS client for sending fraud alert notifications.

In development mode, messages are logged instead of being sent through
the Twilio API, to avoid consuming SMS credits during testing.
"""

import logging
import re
from typing import Optional

from config import get_settings

logger = logging.getLogger("fraud_guardian.twilio")


def format_nigerian_phone(phone: str) -> str:
    """
    Convert various Nigerian phone number formats to E.164 (+234...).

    Accepted input formats:
        - 08012345678  (local with leading zero)
        - 8012345678   (local without leading zero)
        - 2348012345678 (international without +)
        - +2348012345678 (already E.164)
        - 0 801 234 5678 (with spaces)

    Args:
        phone: Phone number string in any common Nigerian format.

    Returns:
        Phone number in E.164 format (+234XXXXXXXXXX).

    Raises:
        ValueError: If the phone number cannot be parsed.
    """
    # Strip whitespace, dashes, and parentheses
    cleaned = re.sub(r"[\s\-\(\)]+", "", phone.strip())

    # Already in E.164 with country code
    if cleaned.startswith("+234"):
        if len(cleaned) == 14:
            return cleaned
        raise ValueError(f"Invalid Nigerian phone number length: {phone}")

    # International format without +
    if cleaned.startswith("234") and len(cleaned) == 13:
        return f"+{cleaned}"

    # Local format with leading zero (e.g. 08012345678)
    if cleaned.startswith("0") and len(cleaned) == 11:
        return f"+234{cleaned[1:]}"

    # Local format without leading zero (e.g. 8012345678)
    if len(cleaned) == 10 and cleaned[0] in "789":
        return f"+234{cleaned}"

    raise ValueError(
        f"Unable to parse Nigerian phone number: '{phone}'. "
        "Expected formats: 08XXXXXXXXX, +234XXXXXXXXXX, 234XXXXXXXXXX, or 8XXXXXXXXX"
    )


class TwilioAlertClient:
    """Client for sending SMS alerts via Twilio."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        """
        Initialize the Twilio SMS client.

        Args:
            account_sid: Twilio Account SID.
            auth_token: Twilio Auth Token.
            from_number: Twilio phone number to send from (E.164 format).
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self._settings = get_settings()

    def send_sms(self, to: str, message: str) -> dict:
        """
        Send an SMS message to a Nigerian phone number.

        In development mode, the message is logged but not actually sent
        through the Twilio API. In production, Twilio's REST client is used.

        Args:
            to: Recipient phone number (any Nigerian format).
            message: SMS body text.

        Returns:
            Dict with keys:
                - status: 'sent', 'logged', or 'failed'
                - sid: Twilio message SID (None if not sent)
                - error: Error message string (None on success)
        """
        try:
            formatted_to = format_nigerian_phone(to)
        except ValueError as exc:
            logger.error("Invalid phone number '%s': %s", to, exc)
            return {"status": "failed", "sid": None, "error": str(exc)}

        # Development mode: log instead of sending
        if self._settings.is_development:
            logger.info(
                "[DEV MODE] SMS to %s from %s:\n%s",
                formatted_to,
                self.from_number,
                message,
            )
            return {"status": "logged", "sid": None, "error": None}

        # Production mode: send via Twilio
        try:
            from twilio.rest import Client as TwilioRestClient

            client = TwilioRestClient(self.account_sid, self.auth_token)
            twilio_message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=formatted_to,
            )

            logger.info(
                "SMS sent to %s: SID=%s, status=%s",
                formatted_to,
                twilio_message.sid,
                twilio_message.status,
            )
            return {
                "status": "sent",
                "sid": twilio_message.sid,
                "error": None,
            }

        except ImportError:
            logger.error("Twilio library not installed. Install with: pip install twilio")
            return {
                "status": "failed",
                "sid": None,
                "error": "Twilio library not installed",
            }
        except Exception as exc:
            logger.error("Failed to send SMS to %s: %s", formatted_to, exc)
            return {
                "status": "failed",
                "sid": None,
                "error": str(exc),
            }


def get_twilio_client() -> TwilioAlertClient:
    """
    Factory function to create a TwilioAlertClient from application settings.

    Returns:
        Configured TwilioAlertClient instance.
    """
    settings = get_settings()
    return TwilioAlertClient(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        from_number=settings.twilio_phone_number,
    )
