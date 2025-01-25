# Start from a slim Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app

# Create and switch to the application directory
WORKDIR $APP_HOME

# Copy all project files into the container
COPY . ./

# Install Python dependencies
# Make sure your requirements.txt includes selenium, pytz, etc.
RUN pip install --no-cache-dir -r requirements.txt

# Expose 8080 (the standard Cloud Run port)
EXPOSE 8080

CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]