from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathowatch import load_model_system, detect_location

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading ML system...")

model, dataset, features, b2, b3, b4, b8 = load_model_system()

print("System ready")

@app.get("/detect")
def detect(lat: float, lon: float):

    result = detect_location(
        model,
        dataset,
        features,
        b2,
        b3,
        b4,
        b8,
        lat,
        lon
    )

    return result