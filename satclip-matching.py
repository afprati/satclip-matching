import geopandas as gpd
import torch
# pip install timm
# pip install torchgeo
# pip install albumentations

import sys
sys.path.append("C:/Users/miame/OneDrive/Backups/Documents/GitHub/satclip/satclip")  # path to the cloned repo
from model import SatCLIP
from location_encoder import *
from load import get_satclip


from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, bbox_to_dimensions, BBox
import numpy as np
from scipy.spatial.distance import cdist
import config as c

# Load the shapefile
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.head(5)
    return gdf



# Initialize SatCLIP model
def load_satclip():
    model = SatCLIP(
        embed_dim=512,
        image_resolution=224, 
        in_channels=13, 
        vision_layers=4, 
        vision_width=768, 
        vision_patch_size=32, # Image encoder
        le_type='sphericalharmonics', 
        pe_type='siren', 
        legendre_polys=10, 
        frequency_num=16, 
        max_radius=360, 
        min_radius=1, 
        harmonics_calculation='analytic'  # Location encoder
        )
    model.eval()
    return model


# Configure Sentinel Hub API
config = SHConfig()
config.instance_id = c.config_id
config.sh_client_id = c.client_id
config.sh_client_secret = c.client_secret

# Fetch Sentinel-2 image for a given bounding box
def fetch_satellite_image(bbox_coords):
    bbox = BBox(bbox_coords, crs="EPSG:4326")
    resolution = 10  # meters per pixel
    size = bbox_to_dimensions(bbox, resolution=resolution)
    size = (min(size[0], 2500), min(size[1], 2500)) # ensure size does not exceed Sentinel Hub's limits
    
    request = SentinelHubRequest(
        evalscript="""
        // SentinelHub evalscript to get true color image
        function setup() {
            return {
                input: [{bands: ["B04", "B03", "B02"], resolution: 10}],
                output: {bands: 3, sampleType: "UINT8"}
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2023-01-01", "2023-12-31")
        )],
        responses=[SentinelHubRequest.output_response("default", "png")],
        bbox=bbox,
        size=size,
        config=config
    )
    response = request.get_data()
    print("Here is the response:")
    print(response)
    image = Image.open(BytesIO(response[0]))
    return image

# Preprocess satellite image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Generate SatCLIP embeddings for grid cells
def generate_embeddings(gdf, model):
    embeddings = {}
    for _, row in gdf.iterrows():
        bbox_coords = row['geometry'].bounds  # Extract bounding box from geometry
        image = fetch_satellite_image(bbox_coords)
        image_tensor = preprocess_image(image)
        
        with torch.no_grad():
            embedding = model.encode_image(image_tensor).numpy()
        embeddings[row['grid_id']] = embedding
    return embeddings

# Perform one-to-one matching based on cosine similarity
def match_treated_to_untreated(gdf, embeddings):
    treated = gdf[gdf['treated'] == 1]
    untreated = gdf[gdf['treated'] == 0]
    
    treated_ids = treated['grid_id'].tolist()
    untreated_ids = untreated['grid_id'].tolist()
    
    treated_embeddings = np.vstack([embeddings[grid_id] for grid_id in treated_ids])
    untreated_embeddings = np.vstack([embeddings[grid_id] for grid_id in untreated_ids])
    
    distances = cdist(treated_embeddings, untreated_embeddings, metric='cosine')
    
    matches = {}
    used_indices = set()
    for i, treated_id in enumerate(treated_ids):
        min_index = np.argmin(distances[i])
        while min_index in used_indices:  # Ensure one-to-one matching
            distances[i, min_index] = np.inf
            min_index = np.argmin(distances[i])
        
        matches[treated_id] = untreated_ids[min_index]
        used_indices.add(min_index)
    
    return matches

# Main execution function
def main(shapefile_path):
    gdf = load_shapefile(shapefile_path)
    model = load_satclip()
    embeddings = generate_embeddings(gdf, model)
    matches = match_treated_to_untreated(gdf, embeddings)
    return matches

# Example usage
matches = main("grids_shapefile.shp")
