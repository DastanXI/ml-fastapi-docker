# ML model deployment with FastAPI and Docker

A small machine learning service. A scikit-learn classifier is trained on the Iris dataset,
served through a FastAPI application, and packaged into a Docker image.

The model predicts the flower species (`setosa`, `versicolor`, `virginica`) from four measurements.

## Project structure

```
ml-fastapi-docker/
├── main.py            FastAPI app (loads the model, serves predictions)
├── train.py           trains the model and saves it to model.joblib
├── model.joblib       saved trained model (artifact loaded by the API)
├── requirements.txt   Python dependencies (pinned versions)
├── Dockerfile         container instructions
├── .dockerignore      files that are not copied into the image
├── tests/test_api.py  automated API tests
└── README.md          this file
```

## API endpoints

| Method | Path       | Description |
|--------|------------|-------------|
| GET    | `/`        | Returns `{"message": "ML API is running"}` |
| POST   | `/predict` | Takes the four features, returns the predicted species and class probabilities |
| GET    | `/health`  | Extra endpoint: service and model status (used by the Docker health check) |
| GET    | `/docs`    | Interactive Swagger documentation |

Example request body for `/predict` (all values in centimetres):

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

Example response:

```json
{"prediction": "setosa", "class_id": 0, "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}}
```

Invalid input (missing field, negative number, wrong type) returns HTTP 422 with an explanation.

## Step 1. Train the model

```bash
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
python train.py
```

This prints the test accuracy and creates `model.joblib`. A trained `model.joblib` is already included,
so you can skip this step, but running it again is safe.

## Step 2 and 3. Run the API locally

```bash
uvicorn main:app --reload
```

Open these in the browser:

- http://127.0.0.1:8000/ shows the running message
- http://127.0.0.1:8000/docs shows Swagger. Open `POST /predict`, press **Try it out**, then **Execute**.

Test from the terminal:

```bash
# macOS / Linux / Git Bash
curl http://127.0.0.1:8000/
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

```powershell
# Windows PowerShell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Run the automated tests (optional):

```bash
pip install pytest httpx
pytest
```

## Step 4 and 5. Build the Docker image

Docker Desktop (or the Docker engine) must be running. From the project folder:

```bash
docker build -t ml-fastapi-app .
```

The build must finish without errors. Check that the image exists with `docker images`.

## Step 6. Run the container

```bash
docker run -d --name ml-api -p 8000:8000 ml-fastapi-app
```

`-p 8000:8000` maps port 8000 of your computer to port 8000 inside the container.
Then test again, exactly as in the local run:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs
- `POST /predict` (curl or Swagger commands above)

Useful commands:

```bash
docker ps                 # is the container running (and healthy)?
docker logs ml-api        # server logs
docker stop ml-api        # stop it
docker rm ml-api          # remove it
```

## How it works

1. `train.py` trains a Random Forest and saves it with `joblib` together with the feature names and class names.
2. `main.py` loads `model.joblib` once at startup, validates each request with Pydantic, and calls `predict` and `predict_proba`.
3. The `Dockerfile` starts from `python:3.12-slim`, installs the pinned dependencies, copies the project, and runs Uvicorn on `0.0.0.0:8000`.
   Dependencies are installed before copying the code, so rebuilding after a code change is fast.

## Troubleshooting

- **Port already in use:** use another host port, for example `docker run -p 8080:8000 ml-fastapi-app`, then open port 8080.
- **`Cannot connect to the Docker daemon`:** start Docker Desktop and wait until it says it is running.
- **Warning about scikit-learn versions:** the model must be loaded with the same scikit-learn version that trained it.
  The versions are pinned in `requirements.txt`. If you change them, run `python train.py` again.
- **`model.joblib not found`:** run `python train.py` in the project folder.

## Notes

- The Iris test set has only 30 samples, so the reported accuracy is a rough estimate.
- The container runs as a non-root user and has a health check on `/health`.
