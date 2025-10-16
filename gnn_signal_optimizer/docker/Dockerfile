FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && add-apt-repository ppa:sumo/stable -y \
    && apt-get update && apt-get install -y sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install -e .[dev]

EXPOSE 8000
CMD ["uvicorn", "gso.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
