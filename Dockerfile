FROM python:3.11-slim

# Install WeasyPrint system dependencies + fontconfig + wget
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    libgobject-2.0-0 \
    libxml2 \
    libxslt1.1 \
    fontconfig \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install Noto Sans Ethiopic to a standard font directory
RUN mkdir -p /usr/share/fonts/truetype/noto && \
    wget -O /usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf \
    https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSansEthiopic/NotoSansEthiopic-Regular.ttf && \
    fc-cache -fv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd -r portal && useradd -r -g portal portal
USER portal

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]