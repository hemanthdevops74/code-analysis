FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port that Streamlit will run on (default is 8501)
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "code_chat.py", "--server.port=8501"]
