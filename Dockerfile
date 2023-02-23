FROM bitnami/pytorch:latest
LABEL maintainer="Jorge Roman"
# Set the working directory
WORKDIR /app
# Copy the app code
COPY app.py /app
COPY test.py /app

# Copy the models directory
COPY models /app/models

# Install any additional packages
RUN pip install -r requirements.txt

# Expose port 5000 
EXPOSE 5000
# Set the environment variable for Flask 
ENV FLASK_APP=app.py
# Run the command to start the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]