FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml config.yaml ./
RUN pip install -e . && pip cache purge

COPY . .

ENTRYPOINT ["python", "main.py"]
