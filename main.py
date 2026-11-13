import hmac
import os
import tempfile
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from analyzer import profile_csv

API_KEY = os.environ["ANALYZER_KEY"]
ALLOWED_SUFFIXES = tuple(
    s.strip().lower() for s in os.environ.get("ALLOWED_FILE_HOSTS", "").split(",") if s.strip()
) 
MAX_BYTES = int(os.environ.get("MAX_FILE_MB", "25")) * 1024 * 1024

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: str
    delimiter: str = ","
    header: bool = True


@app.get("/health")
def health():
    return {"ok": True}


def check_url(url: str) -> None:
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise HTTPException(400, "file link must be https")
    if not ALLOWED_SUFFIXES or not host.endswith(ALLOWED_SUFFIXES):
        raise HTTPException(400, "file host not allowed")


def download(url: str, dest: str) -> None:
    size = 0
    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=False) as r:
        if r.status_code != 200:
            raise HTTPException(404, "could not fetch file")
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "file too large")
                f.write(chunk)


@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_api_key: str = Header(...)):
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "bad key")
    check_url(req.url)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.csv")
        download(req.url, path)
        try:
            return profile_csv(path, req.delimiter, req.header)
        except Exception as e:
            raise HTTPException(422, f"could not analyze file: {e}")