import hmac
import os
import tempfile

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from analyzer import profile_csv

API_KEY = os.environ["ANALYZER_KEY"]
MAX_BYTES = int(os.environ.get("MAX_FILE_MB", "25")) * 1024 * 1024

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    delimiter: str = Form(","),
    header: bool = Form(True),
    x_api_key: str = Header(...)
):
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "bad key")

    # Save the incoming streamed file chunk-by-chunk to a temp file safely
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.csv")
        size = 0
        
        try:
            with open(path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "file too large")
                    f.write(chunk)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(400, f"could not read uploaded file: {e}")

        try:
            return profile_csv(path, delimiter, header)
        except Exception as e:
            raise HTTPException(422, f"could not analyze file: {e}")