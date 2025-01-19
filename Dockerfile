FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app
ENV PORT=5000

# Create and switch to the application directory
WORKDIR $APP_HOME

# Install system dependencies needed for Chrome/Chromium + ChromeDriver
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, but good practice to document)
EXPOSE $PORT

# Run gunicorn server
CMD ["gunicorn", "--bind", ":5000", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
