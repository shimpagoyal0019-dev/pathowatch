from flask import Flask,send_file,jsonify
import pathowatch_pipeline

app = Flask(__name__)

model = None
heatmap = None
# ---------------------------
# Home Route (ADD HERE)
# ---------------------------
@app.route("/")
def home():
    return {
        "message": "PathoWatch API running",
        "routes": [
            "/run_model",
            "/risk_map",
            "/risk_stats"
        ]
    }


# ---------------------------
# Run Model
# ---------------------------

@app.route("/run_model")

def run_model():

    global model,heatmap

    model,heatmap = pathowatch_pipeline.run_pipeline()

    return {"status":"model_run_complete"}


# ---------------------------
# Get Risk Map
# ---------------------------

@app.route("/risk_map")

def risk_map():

    return send_file("risk_map.png",mimetype="image/png")


# ---------------------------
# Risk Statistics
# ---------------------------

@app.route("/risk_stats")

def risk_stats():

    global heatmap

    if heatmap is None:
        return {"error":"model not run"}

    high = (heatmap>0.7).sum()
    medium = ((heatmap>0.4)&(heatmap<=0.7)).sum()
    low = (heatmap<=0.4).sum()

    return jsonify({
        "high_risk_pixels":int(high),
        "medium_risk_pixels":int(medium),
        "low_risk_pixels":int(low)
    })
@app.route("/risk_at_location")
def risk_at_location():

    from flask import request

    global model, heatmap

    if heatmap is None:
        return {"error":"model not run"}

    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    import rasterio
    dataset = rasterio.open("sentinel.tif")

    row, col = dataset.index(lon, lat)

    probability = float(heatmap[row, col])

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "latitude":lat,
        "longitude":lon,
        "probability":probability,
        "risk_level":risk
    }
@app.route("/hotspots")
def hotspots():

    global heatmap

    import numpy as np

    threshold = 0.75

    points = np.argwhere(heatmap > threshold)

    hotspots = []

    for r,c in points[::500]:

        hotspots.append({
            "row":int(r),
            "col":int(c)
        })

    return {"hotspots":hotspots}


# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    app.run(port=5000,debug=True)