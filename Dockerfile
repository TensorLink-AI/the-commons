FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml server.py ./

RUN pip install --no-cache-dir .

VOLUME /data

EXPOSE 8000

# Set API_TOKEN env var at runtime to enable bearer-token auth:
#   docker run -e API_TOKEN=mysecret ...
ENV API_TOKEN=""

ENTRYPOINT ["the-commons"]
CMD ["--transport", "sse", "--port", "8000", "--db", "/data/the_commons.db"]
