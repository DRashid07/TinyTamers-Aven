# Railway image: the API and both web pages for the demo, CPU only.
# The data pipeline (mediapipe, opencv, scikit-learn, matplotlib) is not needed to serve, so only the server
# packages are installed, at the versions pinned in requirements.txt. torch comes from the CPU index:
# the default Linux wheel would pull about 2.5 GB of CUDA libraries.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements.txt)" \
 && grep -E '^(fastapi|uvicorn|python-dotenv|numpy|anthropic|httpx)' requirements.txt > requirements-server.txt \
 && pip install -r requirements-server.txt

# Code plus the files that are not in git but uploaded by `railway up --no-gitignore` (see .railwayignore):
# model/artifacts/model.pt + config.json, data/clips/*.mp4 and web/models/*.task.
COPY . .

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
