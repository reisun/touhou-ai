FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3
WORKDIR /app
COPY bridge.py smoke.py ./
COPY tests ./tests
USER 10001:10001
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
