import rasterio
import numpy as np
import matplotlib.pyplot as plt
import ee
import geemap
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.animation as animation


# ---------------------------
# Download Sentinel Image
# ---------------------------
def download_satellite():

    point = ee.Geometry.Point([77.2090, 28.6139])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    image = collection.first()

    bands = image.select(["B2", "B3", "B4", "B8"])

    geemap.ee_export_image(
        bands,
        filename="sentinel.tif",
        scale=10,
        region=point.buffer(5000)
    )

    return "sentinel.tif"


# ---------------------------
# Load Bands
# ---------------------------
def load_bands(filename):

    dataset = rasterio.open(filename)

    b2 = dataset.read(1)
    b3 = dataset.read(2)
    b4 = dataset.read(3)
    b8 = dataset.read(4)

    return dataset, b2, b3, b4, b8


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(b2, b3, b4, b8):

    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-10)

    spectral_slope = (b8 - b2) / (b8 + b2 + 1e-10)

    absorption_depth = np.minimum.reduce([b2, b3, b4, b8])

    spectral_variance = np.var(np.stack([b2, b3, b4, b8], axis=0), axis=0)

    spectral_gradient = np.gradient(
        np.stack([b2, b3, b4, b8], axis=0), axis=0
    )[0]

    features = np.stack([
        b2, b3, b4, b8,
        ndvi, ndwi,
        absorption_depth,
        spectral_variance,
        spectral_slope,
        spectral_gradient
    ], axis=-1)

    return features, ndvi


# ---------------------------
# Train ML Model
# ---------------------------
def train_model(features, ndvi):

    psi = (1 - ndvi)

    X = features.reshape(-1, 10)

    labels = (psi > np.percentile(psi, 75)).astype(int)

    y = labels.flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nModel Evaluation:\n")
    print(classification_report(y_test, pred))

    return model


# ---------------------------
# Generate Heatmap
# ---------------------------
def generate_heatmap(model, features, b2):

    X = features.reshape(-1, 10)

    probabilities = model.predict_proba(X)[:, 1]

    heatmap = probabilities.reshape(b2.shape)

    plt.imshow(heatmap, cmap="hot")
    plt.title("Pathogen Detection Heatmap")
    plt.colorbar()
    plt.show()

    plt.imsave("heatmap.png", heatmap, cmap="hot")

    return heatmap


# ---------------------------
# Detect Pathogen at Location
# ---------------------------
def detect_location(model, dataset, features):

    geolocator = Nominatim(user_agent="pathowatch")

    location_name = input("\nEnter location: ")

    location = geolocator.geocode(location_name)

    if location is None:
        print("Location not found")
        return None, None

    lat = location.latitude
    lon = location.longitude

    print("Latitude:", lat)
    print("Longitude:", lon)

    row, col = dataset.index(lon, lat)

    pixel = features[row, col].reshape(1, -1)

    probability = model.predict_proba(pixel)[0][1]

    print("\nPathogen probability:", probability)

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    print("Risk Level:", risk)

    return row, col


# ---------------------------
# Spectral Graph
# ---------------------------
def spectral_graph(b2, b3, b4, b8, row, col):

    values = [
        b2[row, col],
        b3[row, col],
        b4[row, col],
        b8[row, col]
    ]

    bands = ["Blue", "Green", "Red", "NIR"]

    plt.figure(figsize=(6, 4))

    plt.plot(bands, values, marker="o", linewidth=2)

    plt.title("Spectral Absorption Signature")

    plt.xlabel("Spectral Band")

    plt.ylabel("Reflectance")

    plt.grid()

    plt.show()


# ---------------------------
# Spread Animation
# ---------------------------
def spread_animation(heatmap):

    fig = plt.figure()

    frames = []

    for i in range(5):
        frames.append([plt.imshow(heatmap, cmap="hot", animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=800)

    ani.save("pathogen_spread.gif", writer="pillow")

    print("\nAnimation saved as pathogen_spread.gif")


# ---------------------------
# Main Program
# ---------------------------
def main():

    ee.Authenticate()
    ee.Initialize()

    filename = download_satellite()

    dataset, b2, b3, b4, b8 = load_bands(filename)

    features, ndvi = extract_features(b2, b3, b4, b8)

    model = train_model(features, ndvi)

    heatmap = generate_heatmap(model, features, b2)

    row, col = detect_location(model, dataset, features)

    if row is not None:
        spectral_graph(b2, b3, b4, b8, row, col)

    spread_animation(heatmap)


# ---------------------------
# Run Script
# ---------------------------
if __name__ == "__main__":
    main()