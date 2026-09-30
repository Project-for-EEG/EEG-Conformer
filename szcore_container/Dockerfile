# syntax=docker/dockerfile:1
#
# Seizure detector for the SzCORE evaluation.
#
# Built from config/template.Dockerfile in esl-epfl/szcore, with two changes.
# The template's apt-get layer is dropped: as committed it ends a line with
# "unzip &&" and no continuation, which is a shell syntax error, and nothing
# here needs wget, tar, git or unzip. And torch comes from the CPU wheel index,
# because whether the evaluation host has a GPU is not documented on the
# current submission page -- the figures that mention one belong to the 2025
# challenge -- so this must run without one.
#
# Nothing is downloaded at run time. The weights are baked in.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# The image is built on Windows, where the default codepage makes some source
# distributions fail to read their own UTF-8 metadata.
ENV PYTHONUTF8=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install --no-cache-dir -r requirements.txt

COPY algo/ ./algo/
COPY *.py ./
COPY weights/ ./weights/

# szcore_run reads this, so the checkpoint location is not hard-coded in code
# that is shared with the repository.
ENV CHECKPOINT=/app/weights/szcore_deploy.pt

USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

CMD python3 -m algo "/data/${INPUT}" "/output/${OUTPUT}"
