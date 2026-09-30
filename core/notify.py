"""
Shared Telegram notification helper -- not specific to any one machine, same spirit as
core/db.py and core/settings.py. Credentials come from pdm-services/.env
(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID, loaded by core.settings on import), read lazily
(at send time, not at import) so a module that only needs this file for its constants
stays importable without a configured .env -- same reasoning as settings.py's db_url().

The real IoT server already sends its production alerts to Telegram; this is meant for
local dev/testing against a SEPARATE dev bot, so a test send never lands in a real
production alert chat. Point TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID at whichever bot/chat
is appropriate for the environment this runs in.
"""
import os

import requests

from . import settings  # noqa: F401  (import triggers settings.load_dotenv(...) as a side effect)

TIMEOUT_S = 10


def send_telegram(text, parse_mode=None):
    """Best-effort: returns True/False, never raises -- a notification failure should
    never take down whatever caller triggered it (matches band_drift_watch's dashboard
    banner, which already swallows its own exceptions for the same reason). Prints a
    warning instead of silently doing nothing when unset/failed, so a misconfigured
    .env is noticeable in the console rather than a mystery.

    `parse_mode`: None (default) sends plain text -- always safe, use this unless the
    caller has actually checked its text. Pass "Markdown" ONLY if every interpolated
    field that isn't a fixed/known-safe string is wrapped in `backticks` (a code span):
    Telegram's legacy Markdown treats a lone _ / * / ` as an entity delimiter with NO
    escaping, and it fails two different ways depending on how many of the character
    appear -- confirmed 2026-09-29 (see project memory "toe-lasting rul trigger
    telegram template"): an ODD count (e.g. "thermocouple_failure", one `_`) makes the
    WHOLE send fail with a 400 "can't parse entities"; an EVEN count (e.g.
    "motor_winding_failure", two `_`) does NOT fail but silently mangles the text
    instead (Telegram consumes the underscores as an unintended italic span --
    "motor_winding_failure" is delivered as "motorwindingfailure" with "winding"
    italicized). A backtick-wrapped code span is not scanned for either case."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("notify.send_telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set in .env -- skipped")
        return False
    try:
        data = {"chat_id": chat_id, "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            timeout=TIMEOUT_S,
        )
        if not resp.ok:
            print(f"notify.send_telegram: Telegram API returned {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"notify.send_telegram: request failed -- {e}")
        return False
