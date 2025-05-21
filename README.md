# Project Name: Persona

This project uses FastAPI for the web framework and Celery for handling background tasks, all containerized with Docker and managed by Docker Compose.

## Configuration

1.  **Environment Variables:**
    This project requires certain environment variables to be set. Create a `.env` file in the root of the project by copying `.env.example` (if you provide one) or by creating it manually.

    Example `.env` file content:
    ```env
    # .env

    # Your Upstash Redis URL (including username, password, host, and port)
    # Example: rediss://username:password@your-upstash-instance.upstash.io:port
    REDIS_URL_UPSTASH=your_full_upstash_redis_url_with_credentials

    SUPABASE_URL=your_supabase_url
    SUPABASE_SERVICE_KEY=your_supabase_service_key
    SUPABASE_JWT_SECRET=your_supabase_jwt_secret
    GOOGLE_API_KEY=your_google_api_key

    FLASK_ENV=development # Set to 'production' for production builds
    ```
    **Replace the placeholder values with your actual credentials and configuration.**

## Building and Running the Application with Docker Compose

Follow these steps to build the Docker images and run the application services:

1.  **Navigate to the Project Directory:**
    Open your terminal and change to the root directory of this project (where `docker-compose.yml` is located).

2.  **Build and Start the Services:**
    Run the following command:
    ```bash
    docker-compose up --build
    ```
    *   This command will start:
        *   The FastAPI web service (typically accessible at `http://localhost:5000`).
        *   The Celery worker service, which will connect to your Upstash Redis instance and process background tasks.
    *   You will see interleaved logs from both services in your terminal.

3.  **Accessing the Application:**
    Once the services are up and running, you should be able to access your FastAPI application by opening a web browser or using a tool like Postman/curl to `http://localhost:5000` (or whichever port you have configured and mapped).

## Common Docker Compose Commands

*   **Running in Detached Mode:**
    To run the services in the background (detached mode), use the `-d` flag:
    ```bash
    docker-compose up --build -d
    ```

*   **Stopping the Services:**
    To stop the running services:
    *   If running in the foreground (`docker-compose up`), press `Ctrl+C` in the terminal.
    *   If running in detached mode, or from another terminal:
        ```bash
        docker-compose down
        ```
        This command stops and removes the containers, networks, and volumes (unless specified otherwise). To just stop without removing: `docker-compose stop`.

*   **Rebuilding an Image for a Specific Service:**
    If you've only made code changes that affect one service and its image needs rebuilding:
    ```bash
    docker-compose build web      # Rebuild only the 'web' service image
    docker-compose build worker   # Rebuild only the 'worker' service image
    # Then start services again:
    docker-compose up -d --no-deps web # Start/recreate only web, don't start its dependencies
    ```
    Often, just running `docker-compose up --build` is simpler if you're unsure.

*   **Forcing a Rebuild of All Images:**
    ```bash
    docker-compose build --no-cache
    # Then
    docker-compose up
    ```

