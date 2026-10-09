"""Join requests from the home page (web/home.html, "Qoşulma sorğusu"): GET /join-token, POST /join-request,
GET /join-requests.

Owner: C (Backend/LLM). Aven has no accounts and no log-in (no personal access, unlike a client panel): a request only
asks the team to call back, like the join form on Clopos's site. The spam guards are Project V7's sign-up and contact
endpoints: a honeypot field ("website") that gets a fake success, a start time signed by the server (the form must be
open for MIN_FILL_S seconds) and fixed-window rate limits (CLIENT_LIMIT per client, GLOBAL_LIMIT for everyone).

Accepted requests are appended as JSON lines to data/join_requests/requests.jsonl (JOIN_REQUESTS_PATH overrides it,
relative to sign-assistant/). The folder is in .gitignore and .railwayignore because it holds names and phone numbers;
on Railway it lasts until the next deploy unless JOIN_REQUESTS_PATH points to a volume. The team reads the requests with
    curl -H "Authorization: Bearer $JOIN_ADMIN_TOKEN" https://<host>/join-requests
and without JOIN_ADMIN_TOKEN that endpoint answers 404. JOIN_FORM_SECRET signs the start time; when it is empty a random
key is made at start-up, so a form opened before a restart asks for a page reload.

Answers: 200 {"ok": true, "message"}; otherwise {"ok": false, "error": <Azerbaijani text>} with "field" (422, the
field to fix) or "code" (400 "token" / "too_fast"), 429 with Retry-After, 503 when the file cannot take more.
"""
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "join_requests" / "requests.jsonl"
MIN_FILL_S = 3  # V7 ContactEndpoints.MinFormFillSeconds
TOKEN_MAX_AGE_S = 2 * 60 * 60
CLIENT_LIMIT = (5, 300)  # V7's "contact" policy: 5 requests per 300 s
GLOBAL_LIMIT = (60, 300)  # backstop, also against a client that changes its forwarded address
MAX_STORE_BYTES = 5_000_000

SECTORS = {"government": "Dövlət xidməti", "service": "Bank və xidmət", "education": "Təhsil",
           "community": "İcma və QHT", "other": "Digər"}
PURPOSES = {"try": "Sınamaq istəyirəm", "support": "Dəstək olmaq istəyirəm"}
PHONE_RE = re.compile(r"^(?:\+994|0)([1-9]\d{8})$")  # Azerbaijani mobile or landline: +994 / 0 and 9 digits
SUCCESS = "Təşəkkür edirik! Sorğunuzu qəbul etdik, komandamız qısa müddətdə sizinlə əlaqə saxlayacaq."

log = logging.getLogger("uvicorn.error")
router = APIRouter()
_lock = threading.Lock()  # sync endpoints run in a thread pool
_random_secret = secrets.token_bytes(32)


class JoinRequest(BaseModel):
    name: str = Field("", max_length=200)
    phone: str = Field("", max_length=40)
    organization: str = Field("", max_length=300)
    sector: str = Field("", max_length=40)
    purpose: str = Field("", max_length=40)
    consent: bool = False
    website: str = Field("", max_length=300)  # honeypot: hidden from people, so only bots fill it
    form_token: str = Field("", max_length=200)


class FixedWindow:
    """Requests per key in fixed windows of `seconds`, like ASP.NET Core's FixedWindowRateLimiter in V7."""

    def __init__(self, limit, seconds):
        self.limit, self.seconds = limit, seconds
        self.windows = {}  # key -> (window start, count)

    def hit(self, key, now):
        """Counts one request. Returns 0 if it is allowed, else the seconds until the window ends."""
        start, count = self.windows.get(key, (now, 0))
        if now - start >= self.seconds:
            start, count = now, 0
        if count >= self.limit:
            return start + self.seconds - now
        self.windows[key] = (start, count + 1)
        if len(self.windows) > 10_000:  # drop finished windows so the dict cannot grow without bound
            self.windows = {k: v for k, v in self.windows.items() if now - v[0] < self.seconds}
        return 0.0


_per_client = FixedWindow(*CLIENT_LIMIT)
_overall = FixedWindow(*GLOBAL_LIMIT)


class Invalid(Exception):
    def __init__(self, field, message):
        super().__init__(message)
        self.field, self.message = field, message


def _now():
    return time.time()


def reset_limits():
    """Forget all rate-limit windows (tests)."""
    with _lock:
        _per_client.windows.clear()
        _overall.windows.clear()


def _secret():
    return os.getenv("JOIN_FORM_SECRET", "").encode("utf-8") or _random_secret


def _sign(value):
    return hmac.new(_secret(), value.encode("ascii"), hashlib.sha256).hexdigest()


def issue_token(now=None):
    """'<issued ms>.<HMAC-SHA256>': V7's FormTimestamp, signed instead of encrypted (the time is not a secret)."""
    issued = str(int((_now() if now is None else now) * 1000))
    return f"{issued}.{_sign(issued)}"


def token_age(token, now=None):
    """Seconds since the token was issued; None if it is missing, forged or older than TOKEN_MAX_AGE_S."""
    issued, _, signature = (token or "").partition(".")
    if not (issued.isascii() and issued.isdigit()):
        return None
    if not hmac.compare_digest(signature.encode("utf-8"), _sign(issued).encode("ascii")):
        return None
    age = (_now() if now is None else now) - int(issued) / 1000
    return None if age > TOKEN_MAX_AGE_S else max(age, 0.0)


def normalize_phone(raw):
    """'+994 50 123 45 67' / '050-123-45-67' / '(012) 310 22 66' -> '+994501234567'; None if not Azerbaijani."""
    match = PHONE_RE.match(re.sub(r"[\s\-()]", "", raw or ""))
    return f"+994{match.group(1)}" if match else None


def _clean(text):
    return " ".join((text or "").split())


def build_record(body):
    """The line stored for a valid request. Raises Invalid(field, message) with the first problem."""
    name = _clean(body.name)
    if not 3 <= len(name) <= 80 or not any(ch.isalpha() for ch in name):
        raise Invalid("name", "Ad və soyadınızı yazın (3–80 simvol, ən azı 3 hərf).")
    phone = normalize_phone(body.phone)
    if phone is None:
        raise Invalid("phone", "Telefon +994 və ya 0 ilə başlayan Azərbaycan nömrəsi olmalıdır, məsələn +994 50 123 45 67.")
    organization = _clean(body.organization)
    if len(organization) > 100:
        raise Invalid("organization", "Qurumun adı ən çox 100 simvol ola bilər.")
    if body.sector not in SECTORS:
        raise Invalid("sector", "Sahəni seçin.")
    if body.purpose not in PURPOSES:
        raise Invalid("purpose", "Müraciət məqsədini seçin.")
    if not body.consent:
        raise Invalid("consent", "Davam etmək üçün razılığınızı qeyd edin.")
    return {
        "id": uuid.uuid4().hex[:12],
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "phone": phone,
        "organization": organization,
        "sector": body.sector,
        "purpose": body.purpose,
    }


def store_path():
    path = Path(os.getenv("JOIN_REQUESTS_PATH") or DEFAULT_PATH)
    return path if path.is_absolute() else ROOT / path


def append(record):
    """Adds one JSON line. Returns False if the file already holds MAX_STORE_BYTES (nothing is written then)."""
    path = store_path()
    with _lock:
        if path.exists() and path.stat().st_size >= MAX_STORE_BYTES:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def read_all():
    path = store_path()
    if not path.exists():
        return []
    with _lock:
        lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def client_key(request):
    """The client's address. Behind Railway's proxy that is the last X-Forwarded-For entry, the one the proxy added;
    earlier entries come from the browser and are ignored. A spoofed address still lands in GLOBAL_LIMIT."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _fail(status, error, field=None, code=None, headers=None):
    body = {"ok": False, "error": error}
    if field:
        body["field"] = field
    if code:
        body["code"] = code
    return JSONResponse(body, status_code=status, headers=headers)


@router.get("/join-token")
def join_token():
    return JSONResponse({"token": issue_token()}, headers={"Cache-Control": "no-store"})


@router.post("/join-request")
def join_request(body: JoinRequest, request: Request):
    now = _now()
    with _lock:
        wait = _per_client.hit(client_key(request), now) or _overall.hit("*", now)
    if wait:
        return _fail(429, "Çox sayda sorğu göndərildi. Bir neçə dəqiqədən sonra yenidən cəhd edin.",
                     headers={"Retry-After": str(math.ceil(wait))})
    if body.website.strip():
        return {"ok": True, "message": SUCCESS}  # a bot filled the hidden field: it sees success, nothing is kept
    age = token_age(body.form_token, now)
    if age is None:
        return _fail(400, "Forma sessiyası tapılmadı. Səhifəni yeniləyib yenidən cəhd edin.", code="token")
    if age < MIN_FILL_S:
        return _fail(400, "Forma çox tez göndərildi. Bir neçə saniyə gözləyib yenidən cəhd edin.", code="too_fast")
    try:
        record = build_record(body)
    except Invalid as problem:
        return _fail(422, problem.message, field=problem.field)
    try:
        stored = append(record)
    except OSError as error:
        log.error("join request not stored: %s", error)  # never log the request itself: it holds personal data
        stored = False
    if not stored:
        return _fail(503, "Sorğu qəbulu müvəqqəti dayandırılıb. Bir az sonra yenidən cəhd edin.")
    log.info("join request %s stored (%s, %s)", record["id"], record["sector"], record["purpose"])
    return {"ok": True, "message": SUCCESS}


@router.get("/join-requests")
def join_requests(authorization: str = Header("")):
    expected = os.getenv("JOIN_ADMIN_TOKEN", "")
    if not expected:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    scheme, _, given = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip().encode("utf-8"), expected.encode("utf-8")):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
    records = read_all()
    return JSONResponse({"count": len(records), "requests": records[::-1]}, headers={"Cache-Control": "no-store"})
