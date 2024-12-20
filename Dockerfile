# Use the official Python 3.12 slim image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    libopenblas-dev \
    unzip \
    liblapack-dev \
    python3-dev \
    gfortran \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt /app/

# Install a specific version of numpy before other dependencies
# This resolves issues related to deprecated API in gensim
RUN apt-get update && apt-get install -y python3-distutils
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy==1.22.0

# Install Python dependencies
RUN pip install -r requirements.txt


# Download, unzip GloVe, and remove the zip file
RUN curl -O http://nlp.stanford.edu/data/glove.6B.zip \
&& unzip glove.6B.zip \
&& rm glove.6B.zip 

#Remove unnecessary files
RUN rm glove.6B.300d.txt glove.6B.50d.txt glove.6B.200d.txt

# Copy the rest of the application code
COPY . /app

# Expose the default Streamlit port
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
