FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
# scorecard/ (including its artifacts/ subfolder) must contain the pre-built model files --
# this image serves the scoring API only, it does not run the training pipeline. Run the
# full pipeline (warehouse build through scorecard/*.py) outside the container first, so
# scorecard/artifacts/*.pkl exist to be copied in below.
COPY scorecard/ scorecard/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
