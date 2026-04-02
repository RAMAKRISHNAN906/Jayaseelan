FROM python:3.11-slim

# System libraries required by opencv-python-headless and mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads app/static/uploads app/static/processed

EXPOSE 8080

CMD gunicorn main:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
