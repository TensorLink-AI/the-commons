FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml server.py ./

RUN pip install --no-cache-dir .

VOLUME /data

EXPOSE 8000

ENTRYPOINT ["the-commons"]
CMD ["--transport", "sse", "--port", "8000", "--db", "/data/the_commons.db"]
