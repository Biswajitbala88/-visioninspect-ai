FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/srv/.cache/torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml ml
COPY app app
COPY artifacts artifacts
# Bake held-out sample images + detector weights into the image so it runs offline
RUN python -m ml.dataset --out data --per-class 40 --seed 99 \
 && python -c "from app.inference import detector; detector()"
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
