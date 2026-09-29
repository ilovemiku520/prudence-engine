FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=8501
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 appuser \
    && chown appuser:appuser /app
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml
COPY --chown=appuser:appuser docs ./docs
COPY --chown=appuser:appuser README.md LICENSE ./
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8501')+'/_stcore/health',timeout=3)"
CMD ["sh", "-c", "exec python -m streamlit run ui.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
