import geopandas as gpd
import torch
# pip install timm
# pip install torchgeo
# pip install albumentations

import sys
sys.path.append("C:/Users/miame/OneDrive/Backups/Documents/GitHub/satclip/satclip")  # path to the cloned repo
#from model import SatCLIP

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
#from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox
#from shapely.geometry import box
#from io import BytesIO
#from PIL import Image
import requests
from shapely.geometry import Point
from load import get_satclip

device = "cuda" if torch.cuda.is_available() else "cpu"

# download S2-100K dataset
# need to use requests
url = "https://satclip.z13.web.core.windows.net/satclip/index.csv"
response = requests.get(url, verify=False)
if response.status_code == 200:
    with open("S2-100K.csv", "wb") as file:
        file.write(response.content)
    print("S2-100K dataset downloaded")
else:
    print(f"Failed to download dataset. HTTP status code: {response.status_code}")

# load S2-100K dataset
s2_100k_df = pd.read_csv("S2-100K.csv")

# The S2-100K dataset is a dataset of 100,000 multi-spectral satellite images 
# sampled from Sentinel-2 via the Microsoft Planetary Computer. 
# Copernicus Sentinel data is captured between Jan 1, 2021 and May 17, 2023. 
# The dataset is sampled approximately uniformly over landmass and only includes 
# images without cloud coverage. 
# prevents needing to use Sentinel to downlaod images and create new embeddings

# make S2-100K into GeoDataFrame
s2_100k_df['geometry'] = s2_100k_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
s2_100k_gdf = gpd.GeoDataFrame(s2_100k_df, geometry='geometry', crs="EPSG:4326") # same crs as gdf

# load my shapefile with grid cells (created in R)
gdf = gpd.read_file("grids_shapefile.shp")

# getting pre-trained embeddings
model = get_satclip('satclip-vit16-l40.ckpt', device=device)
model.eval()
# ViT models generally perform best for satellite imagery esp in diverse regions like SSA

# spatial join my shapefile to s2_100k_gdf
match = gpd.sjoin(gdf, s2_100k_gdf, how="left", predicate='intersects')
# not one to one join, taking first instance for duplicates
match = match.groupby("grid_id").first().reset_index()


# finding associated embeddings for each grid cells
last_two_columns = match.iloc[:, -2:]

# convert to a numpy array
numpy_array = last_two_columns.values

# Convert the NumPy array to a PyTorch tensor
tensor = torch.tensor(numpy_array)

with torch.no_grad():
  embeddings  = model(tensor.double().to(device)).detach().cpu()
  
# assign embeddings back to dataframe
match['embedding'] = list(embeddings)

# check for NaN embeddings before filtering
print(f"Total NaN embeddings: {match['embedding'].isna().sum()}")

# drop rows with missing embeddings
match = match.dropna(subset=['embedding'])

# separate treated and untreated groups
treated = match[match['treated'] == 1].copy()
untreated = match[match['treated'] == 0].copy()

# debugging check on counts
print(f"Treated: {treated.shape[0]}, Untreated: {untreated.shape[0]}")

# embeddings exist?
if treated.empty or untreated.empty:
    raise ValueError("Either treated or untreated group has no valid embeddings.")


print(f"Total rows: {len(match)}")
print(f"Missing embeddings (None or NaN): {match['embedding'].isna().sum()}")
print(f"Sample embeddings:\n{match['embedding'].head(10)}")  # Print 10 samples


def check_nan_in_tensors(embeddings):
    nan_count = 0
    total_count = len(embeddings)

    for i, emb in enumerate(embeddings):
        if isinstance(emb, torch.Tensor):  # Ensure it's a PyTorch tensor
            emb_np = emb.detach().cpu().numpy()  # Convert to NumPy
            if np.isnan(emb_np).any():  # Check for NaNs inside tensor
                nan_count += 1
                print(f"Embedding {i} contains NaN")  # Print example

    print(f"\nTotal embeddings: {total_count}")
    print(f"Embeddings containing NaNs: {nan_count}")

# run check
check_nan_in_tensors(match['embedding'])

def clean_embeddings(embeddings):
    cleaned = []
    for emb in embeddings:
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
            emb_np = np.nan_to_num(emb_np, nan=0.0)  # Replace NaNs with 0
            cleaned.append(emb_np)
        else:
            continue  # Skip invalid embeddings
    return np.array(cleaned)

# Convert to NumPy arrays with NaNs as 0s
treated_embeddings = clean_embeddings(treated['embedding'])
untreated_embeddings = clean_embeddings(untreated['embedding'])
    


# for all-zero vectors adding small noise**
treated_embeddings[treated_embeddings.sum(axis=1) == 0] += np.random.normal(0, 1e-6, treated_embeddings.shape[1])
untreated_embeddings[untreated_embeddings.sum(axis=1) == 0] += np.random.normal(0, 1e-6, untreated_embeddings.shape[1])


#treated_embeddings = np.nan_to_num(treated_embeddings, nan=0.0)
#untreated_embeddings = np.nan_to_num(untreated_embeddings, nan=0.0)


# check for NaN before cdist**
if np.isnan(treated_embeddings).any():
    raise ValueError("NaN detected in treated embeddings after preprocessing.")
if np.isnan(untreated_embeddings).any():
    raise ValueError("NaN detected in untreated embeddings after preprocessing.")

# cosine similarity
similarity_matrix = 1 - cdist(treated_embeddings, untreated_embeddings, metric='cosine')

# similarity matrix
if np.isnan(similarity_matrix).any():
    raise ValueError("NaN detected in similarity matrix after computation.")

# solve one-to-one assignment
treated_indices, untreated_indices = linear_sum_assignment(similarity_matrix, maximize=True)

# assign matches
treated['matched_untreated_id'] = untreated.iloc[untreated_indices]['grid_id'].values
treated['similarity_score'] = similarity_matrix[treated_indices, untreated_indices]

# save results to csv
treated.to_csv("matched_grid_cells.csv", index=False)
print("Results saved to matched_grid_cells.csv.")

