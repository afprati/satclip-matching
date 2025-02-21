import geopandas as gpd
import torch
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point
from sklearn.decomposition import PCA
import sys
sys.path.append("C:/Users/miame/OneDrive/Backups/Documents/GitHub/satclip/satclip")  # path to the cloned repo
#from model import SatCLIP
#from location_encoder import *
from load import get_satclip

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_s2_100k():
    url = "https://satclip.z13.web.core.windows.net/satclip/index.csv"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        with open("S2-100K.csv", "wb") as file:
            file.write(response.content)
        print("S2-100K dataset downloaded")
    else:
        raise RuntimeError(f"Failed to download dataset. HTTP status code: {response.status_code}")

def load_s2_100k():
    """Loads the S2-100K dataset into a GeoDataFrame."""
    df = pd.read_csv("S2-100K.csv")
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def load_grid_shapefile(filepath):
    """Loads the grid cell shapefile."""
    return gpd.read_file(filepath)

def get_embeddings(model, match):
    """Extracts embeddings for matched grid cells."""
    last_two_columns = match.iloc[:, -2:].values
    tensor = torch.tensor(last_two_columns)
    with torch.no_grad():
        return model(tensor.double().to(device)).detach().cpu()

# cleans embeddings by converting tensors to NumPy arrays and replacing NaNs with 0
def clean_embeddings(embeddings):
    """Cleans embeddings by converting tensors to NumPy arrays and replacing NaNs with 0."""
    cleaned = []
    for emb in embeddings:
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
            emb_np = np.nan_to_num(emb_np, nan=0.0)  # Replace NaNs with 0
            cleaned.append(emb_np)
    return np.array(cleaned)

# matching with cosine similarity; each gridcell can be used up to one time
def perform_matching(treated_embeddings, untreated_embeddings, treated, untreated):
    similarity_matrix = 1 - cdist(treated_embeddings, untreated_embeddings, metric='cosine')
    
    if np.isnan(similarity_matrix).any():
        raise ValueError("NaN detected in similarity matrix after computation.")
    
    treated_indices, untreated_indices = linear_sum_assignment(similarity_matrix, maximize=True)
    treated['matched_untreated_id'] = untreated.iloc[untreated_indices]['grid_id'].values
    treated['similarity_score'] = similarity_matrix[treated_indices, untreated_indices]
    return treated


# summarizes embeddings with PCA, mean, norm, max, min, and variance
def summarize_embeddings(embeddings, grid_ids):
    if embeddings.shape[1] < 3:
        raise ValueError("Embeddings must have at least 3 dimensions for PCA.")

    # compute PCA (3 components)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(embeddings)

    # compute statistics
    mean_val = np.mean(embeddings, axis=1)
    l2_norm = np.linalg.norm(embeddings, axis=1)
    max_val = np.max(embeddings, axis=1)
    min_val = np.min(embeddings, axis=1)
    variance_val = np.var(embeddings, axis=1)

    # create df with extracted features
    summary_df = pd.DataFrame({
        'grid_id': grid_ids.values,
        'PC1': principal_components[:, 0],
        'PC2': principal_components[:, 1],
        'PC3': principal_components[:, 2],
        'Mean': np.mean(embeddings, axis=1),
        'L2_Norm': np.linalg.norm(embeddings, axis=1),
        'Max': np.max(embeddings, axis=1),
        'Min': np.min(embeddings, axis=1),
        'Variance': np.var(embeddings, axis=1)
    })
    return summary_df
    
    return summary_df


def main(grid_shapefile_path):
    download_s2_100k()
    s2_100k_gdf = load_s2_100k()
    gdf = load_grid_shapefile(grid_shapefile_path)
    
    # load model and put in eval mode
    model = get_satclip('satclip-vit16-l40.ckpt', device=device)
    model.eval()
    
    # spatial join
    match = gpd.sjoin(gdf, s2_100k_gdf, how="left", predicate='intersects')
    # some grid cells have multiple images, choosing only the first
    match = match.groupby("grid_id").first().reset_index()
    
    # get embeddings
    match['embedding'] = list(get_embeddings(model, match))
    match = match.dropna(subset=['embedding'])
    
    # separate treated and untreated groups
    treated = match[match['treated'] == 1].copy()
    untreated = match[match['treated'] == 0].copy()
    
    if treated.empty or untreated.empty:
        raise ValueError("either treated or untreated group has no valid embeddings.")
    
    # clean embeddings
    treated_embeddings = clean_embeddings(treated['embedding'])
    untreated_embeddings = clean_embeddings(untreated['embedding'])
    
    # ensure no all-zero vectors
    treated_embeddings[treated_embeddings.sum(axis=1) == 0] += np.random.normal(0, 1e-6, treated_embeddings.shape[1])
    untreated_embeddings[untreated_embeddings.sum(axis=1) == 0] += np.random.normal(0, 1e-6, untreated_embeddings.shape[1])
    
    # matching
    treated = perform_matching(treated_embeddings, untreated_embeddings, treated, untreated)
    
    # save
    treated.to_csv("matched_grid_cells.csv", index=False)
    print("results saved to matched_grid_cells.csv.")
    
    # summarize embeddings
    treated_summary = summarize_embeddings(treated_embeddings, treated['grid_id'])
    untreated_summary = summarize_embeddings(untreated_embeddings, untreated['grid_id'])
    combined_summary = pd.concat([treated_summary, untreated_summary], ignore_index=True)
    
    # save to csv
    combined_summary.to_csv("summarized_embeddings.csv", index=False)
    print("summaries saved to summarized_embeddings.csv.")
    
    return treated

main("grids_shapefile.shp")