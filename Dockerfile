FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:///./recsys.db
EXPOSE 8000
CMD ["uvicorn", "recommend_api:app", "--host", "0.0.0.0", "--port", "8000"]