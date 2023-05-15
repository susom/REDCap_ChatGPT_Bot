# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# INSTALL VIM
RUN apt-get update && apt-get install -y vim

# Copy aliases
COPY .bash_aliases /root/
RUN echo "if [ -f /root/.bash_aliases ]; then . /root/.bash_aliases; fi" >> /root/.bashrc

# Download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# Copy the current directory contents into the container at /app
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 5000

# Set the command to run when the container starts
CMD ["python", "app.py"]


