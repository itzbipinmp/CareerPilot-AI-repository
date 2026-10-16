FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    CAREERPILOT_DB=/data/careerpilot.db \
    APP_ENV=production

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data

EXPOSE 5000
CMD ["sh", "-c", "gunicorn --workers 2 --bind 0.0.0.0:${PORT} app:app"]
