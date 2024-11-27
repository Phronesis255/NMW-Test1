# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Download the SpaCy language model
RUN python -m spacy download en_core_web_sm

# Download, unzip GloVe, and remove the zip file
RUN curl -O http://nlp.stanford.edu/data/glove.6B.zip \
    && unzip glove.6B.zip \
    && rm glove.6B.zip 

#Remove unnecessary files
RUN rm glove.6B.300d.txt glove.6B.50d.txt glove.6B.200d.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the default Streamlit port
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
