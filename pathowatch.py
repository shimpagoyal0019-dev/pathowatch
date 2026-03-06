pip install rasterio numpy matplotlib scikit-learn geopy earthengine-api geemap pillow

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import ee
import geemap
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def download_satellite():

    point = ee.Geometry.Point([77.2090,28.6139])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .filterDate("2023-01-01","2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
    )

    image = collection.first()

    bands = image.select(["B2","B3","B4","B8"])

    geemap.ee_export_image(
        bands,
        filename="sentinel.tif",
        scale=10,
        region=point.buffer(5000)
    )

    return "sentinel.tif"

filename = download_satellite()
dataset = rasterio.open(filename)

b2 = dataset.read(1)
b3 = dataset.read(2)
b4 = dataset.read(3)
b8 = dataset.read(4)

def extract_features(b2,b3,b4,b8):

    ndvi = (b8-b4)/(b8+b4+1e-10)
    ndwi = (b3-b8)/(b3+b8+1e-10)

    spectral_slope = (b8-b2)/(b8+b2+1e-10)

    absorption_depth = np.minimum.reduce([b2,b3,b4,b8])

    spectral_variance = np.var(np.stack([b2,b3,b4,b8],axis=0),axis=0)

    spectral_gradient = np.gradient(np.stack([b2,b3,b4,b8],axis=0),axis=0)[0]

    features = np.stack([
        b2,b3,b4,b8,
        ndvi,ndwi,
        absorption_depth,
        spectral_variance,
        spectral_slope,
        spectral_gradient
    ],axis=-1)

    return features,ndvi,ndwi,absorption_depth,spectral_variance,spectral_slope,spectral_gradient


def train_model(features,ndvi):

    psi = (1-ndvi)

    X = features.reshape(-1,10)

    labels = (psi > np.percentile(psi,75)).astype(int)

    y = labels.flatten()

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print(classification_report(y_test,pred))

    return model
def generate_heatmap(model,features,b2):

    X = features.reshape(-1,10)

    predictions = model.predict(X)

    heatmap = predictions.reshape(b2.shape)

    plt.imshow(heatmap,cmap="hot")
    plt.title("Pathogen Heatmap")
    plt.colorbar()
    plt.show()

    plt.imsave("heatmap.png",heatmap,cmap="hot")

    return heatmap

def detect_location(model,dataset,features):

    geolocator = Nominatim(user_agent="pathowatch")

    location_name = input("Enter location: ")

    location = geolocator.geocode(location_name)

    lat = location.latitude
    lon = location.longitude

    row,col = dataset.index(lon,lat)

    pixel = features[row,col].reshape(1,-1)

    probability = model.predict_proba(pixel)[0][1]

    print("Pathogen probability:",probability)

    def main():

    ee.Authenticate()
    ee.Initialize()

    filename = download_satellite()

    dataset,b2,b3,b4,b8 = load_bands(filename)

    features,ndvi = extract_features(b2,b3,b4,b8)

    model = train_model(features,ndvi)

    generate_heatmap(model,features,b2)

    detect_location(model,dataset,features)











geolocator = Nominatim(user_agent="pathowatch")

location_name = input("Enter location: ")

location = geolocator.geocode(location_name)

lat = location.latitude
lon = location.longitude

print(lat,lon)



dataset = rasterio.open("data/B02.tif")

row,col = dataset.index(lon,lat)

print("Pixel location:",row,col)






values = [
    b2[row,col],
    b3[row,col],
    b4[row,col],
    b8[row,col]
]

print(values)



pixel = np.array([
    b2[row,col],
    b3[row,col],
    b4[row,col],
    b8[row,col],
    ndvi[row,col],
    ndwi[row,col],
    absorption_depth[row,col],
    spectral_variance[row,col]
]).reshape(1,-1)
probability = model.predict_proba(pixel)[0][1]

print("Pathogen probability:", probability)


if probability > 0.7:
    risk = "HIGH"
elif probability > 0.4:
    risk = "MEDIUM"
else:
    risk = "LOW"

print("Risk Level:", risk)


plt.imsave("heatmap.png", heatmap, cmap="hot")






bands = ["Blue","Green","Red","NIR"]

plt.figure(figsize=(6,4))

plt.plot(bands, values, marker="o", linewidth=2)

plt.title("Spectral Absorption Signature")

plt.xlabel("Spectral Band")

plt.ylabel("Reflectance")

plt.grid()

plt.show()

import matplotlib.animation as animation

fig = plt.figure()

frames=[]

for i in range(5):
    frames.append([plt.imshow(heatmap,cmap="hot",animated=True)])

ani = animation.ArtistAnimation(fig,frames,interval=800)

ani.save("pathogen_spread.gif",writer="pillow")