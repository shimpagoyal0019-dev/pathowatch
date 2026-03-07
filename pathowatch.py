import rasterio
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.animation as animation
  
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
# ---------------------------
# Load Sentinel Bands
# ---------------------------
# ---------------------------
# Load Sentinel Bands
# ---------------------------
def load_bands():

    dataset = rasterio.open("Browser_images/B02.tiff")

    b2 = rasterio.open("Browser_images/B02.tiff").read(1)
    b3 = rasterio.open("Browser_images/B03.tiff").read(1)
    b4 = rasterio.open("Browser_images/B04.tiff").read(1)
    b8 = rasterio.open("Browser_images/B08.tiff").read(1)

    return dataset, b2, b3, b4, b8

  


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(b2, b3, b4, b8):

    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)

    ndwi = (b3 - b8) / (b3 + b8 + 1e-10)

    spectral_slope = (b8 - b2) / (b8 + b2 + 1e-10)

    absorption_depth = np.minimum.reduce([b2, b3, b4, b8])

    spectral_variance = np.var(
        np.stack([b2, b3, b4, b8], axis=0), axis=0
    )

    spectral_gradient = np.gradient(
        np.stack([b2, b3, b4, b8], axis=0), axis=0
    )[0]

    features = np.stack([
        b2,
        b3,
        b4,
        b8,
        ndvi,
        ndwi,
        absorption_depth,
        spectral_variance,
        spectral_slope,
        spectral_gradient
    ], axis=-1)

    return features, ndvi


# ---------------------------
# Train Machine Learning Model
# ---------------------------
def train_model(features, ndvi):

    psi = (1 - ndvi)

    X = features.reshape(-1, 10)

    threshold = np.mean(psi)

    labels = (psi > threshold).astype(int)

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
# Generate Pathogen Heatmap
# ---------------------------
def generate_heatmap(model, features, b2):

    X = features.reshape(-1, 10)

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] > 1:
        probabilities = probabilities[:,1]
    else:
        probabilities = probabilities[:,0]

    heatmap = probabilities.reshape(b2.shape)

    # smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=3)

    # detect high risk pixels
    threshold = np.percentile(heatmap, 85)

    high_risk = heatmap > threshold

    coords = np.column_stack(np.where(high_risk))

    if len(coords) > 0:

        clustering = DBSCAN(eps=5, min_samples=20).fit(coords)

        cluster_map = np.zeros_like(heatmap)

        for (r, c), label in zip(coords, clustering.labels_):

            if label != -1:
                cluster_map[r, c] = heatmap[r, c]

        heatmap = gaussian_filter(cluster_map, sigma=5)

    plt.figure(figsize=(8,6))

    plt.imshow(heatmap, cmap="hot")

    plt.title("Pathogen Outbreak Heatmap")

    plt.colorbar(label="Pathogen Probability")

    plt.axis("off")

    plt.show()

    plt.imsave("heatmap.png", heatmap, cmap="hot")

    return heatmap

# ---------------------------
# Location Based Detection
# ---------------------------
def detect_location(model, dataset, features, b2, b3, b4, b8, lat, lon):

    row, col = dataset.index(lon, lat)

    pixel = features[row, col].reshape(1, -1)

    probability = model.predict_proba(pixel)

    if probability.shape[1] > 1:
        probability = probability[0][1]
    else:
        probability = probability[0][0]

    bands = [
        float(b2[row, col]),
        float(b3[row, col]),
        float(b4[row, col]),
        float(b8[row, col])
    ]

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "probability": float(probability),
        "risk": risk,
        "bands": bands
    }

# ---------------------------
# Spectral Signature Graph
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
# Pathogen Spread Animation
# ---------------------------
def spread_animation(heatmap):

    fig = plt.figure()

    frames = []

    for i in range(5):

        frames.append(
            [plt.imshow(heatmap, cmap="hot", animated=True)]
        )

    ani = animation.ArtistAnimation(fig, frames, interval=800)

    ani.save("pathogen_spread.gif", writer="pillow")

    print("\nAnimation saved as pathogen_spread.gif")


# ---------------------------
# Main Program
# ---------------------------
def main():

    dataset, b2, b3, b4, b8 = load_bands()

    features, ndvi = extract_features(b2, b3, b4, b8)

    model = train_model(features, ndvi)

    heatmap = generate_heatmap(model, features, b2)

    print("Model ready")

    return model, dataset, features, b2, b3, b4, b8


def load_model_system():

    dataset, b2, b3, b4, b8 = load_bands()

    features, ndvi = extract_features(b2, b3, b4, b8)

    model = train_model(features, ndvi)

    return model, dataset, features, b2, b3, b4, b8


# ---------------------------
# Run Script
# ---------------------------
if __name__ == "__main__":

    main()