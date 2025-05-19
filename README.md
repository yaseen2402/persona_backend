# Project Name: Your FastAPI & Celery Application

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
    *   `--build`: This flag tells Docker Compose to build the images from your `Dockerfile` before starting the services. You typically need this the first time you run the command or if you've made changes to the `Dockerfile`, `requirements.txt`, or your application code that's copied into the image.
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

*   **Viewing Logs:**
    If running in detached mode, or if you want to see logs from a specific service:
    ```bash
    docker-compose logs -f          # Follow logs for all services
    docker-compose logs -f web      # Follow logs for the 'web' service
    docker-compose logs -f worker   # Follow logs for the 'worker' service
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

## Development Notes

*   **Code Changes:** If you have `volumes: - .:/app` in your `docker-compose.yml` for the `web` and `worker` services, changes to your Python code in the local `./app` directory will be reflected inside the running containers.
    *   For **FastAPI (Uvicorn with reload or Gunicorn managing Uvicorn)**: Uvicorn's reload feature (`uvicorn.run("app:app", reload=True)`) or Gunicorn's reload mechanism (if configured) might pick up these changes automatically. If not, you may need to restart the specific service: `docker-compose restart web`.
    *   For **Celery Worker**: The Celery worker typically needs to be restarted to pick up changes in task definitions or other imported code.
        ```bash
        docker-compose restart worker
        ```

## Troubleshooting

*   **Port Conflicts:** If port `5000` (or any other port you're mapping) is already in use on your host machine, `docker-compose up` will fail. Change the host-side port mapping in `docker-compose.yml` (e.g., `"8000:5000"`) and access the app on the new host port (e.g., `http://localhost:8000`).
*   **Environment Variables:** Ensure all required variables in your `.env` file are correctly set, especially `REDIS_URL_UPSTASH`.
*   **Check Service Logs:** Use `docker-compose logs web` and `docker-compose logs worker` to inspect for errors if services are not starting correctly.

---

This README provides a comprehensive guide. Remember to replace placeholders like "Your FastAPI & Celery Application" with your actual project name. You might also want to add an `.env.example` file to your repository to show the structure of the `.env` file without committing actual secrets.