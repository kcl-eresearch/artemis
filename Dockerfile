FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Playwright's Chromium and its OS dependencies into a shared path
RUN playwright install --with-deps chromium \
    && chmod -R o+rx /opt/playwright

RUN addgroup --system app && adduser --system --ingroup app app

# Copy application
COPY artemis/ ./artemis/

# Expose port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

USER app

# Run the application
CMD ["python", "-m", "artemis.main"]
