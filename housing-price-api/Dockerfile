# Dockerfile
# -----------
# Packages the FastAPI service (housing/api/main.py) + its trained model into one
# self-contained image -- the same idea as a Java "fat JAR" bundling your
# app and its dependencies, except this also includes the OS-level Python
# runtime itself, so it runs identically on ANY machine with Docker.

# Base image: official Python, "slim" variant = smaller than full Python
# image (no unnecessary OS packages), but not so stripped down that
# pip installs break (unlike "alpine", which often needs extra build tools).
FROM python:3.12-slim

# All subsequent commands run from this directory inside the container --
# equivalent to your project root on your own machine.
WORKDIR /app

# Copy ONLY requirements.txt first, before the rest of the code. This is a
# deliberate Docker best practice: Docker caches each instruction as a
# "layer". If requirements.txt hasn't changed, Docker reuses the cached
# "pip install" layer instead of re-running it -- much faster rebuilds
# when you're only changing your Python code, not your dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project code into the image.
COPY . .

# Documents which port the container listens on (informational -- doesn't
# actually publish the port; that happens via `docker run -p` later).
EXPOSE 8000

# The command that runs when the container starts.
# NOTE: --reload is a DEV-ONLY flag (auto-restarts on file changes) and
# should NOT be used in a real container -- there's no live file-editing
# happening inside a shipped container, so we drop it here.
# --host 0.0.0.0 is required: default 127.0.0.1 only listens INSIDE the
# container and would be unreachable from outside it.
CMD ["uvicorn", "housing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
