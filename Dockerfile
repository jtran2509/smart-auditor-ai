FROM python:3.10-slim

WORKDIR /app

# install Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev \
                        && rm -rf /var/lib/apt/lists/*

# Match Hugging Face standard working directory
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# 5.
COPY --chown=user . $HOME/app
WORKDIR $HOME/app

EXPOSE 7860

CMD ["python", "gradio_app.py"]


