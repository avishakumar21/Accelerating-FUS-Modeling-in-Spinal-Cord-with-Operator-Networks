model_type = 'geofno'

import torch
import time
if model_type == 'cnn-deeponet':
    from opnn_cnn import opnn
if model_type == 'vit-deeponet':
    from opnn_transformer import opnn_transformer as opnn
if model_type == 'geofno':
    from geo_fno import GeoFNO
import numpy as np
from dataset_prep import get_paths, TransducerDataset
from torch.utils.data import DataLoader
import os
from utils import plot_prediction, SegmentationVisualizer 


# model config
EXPECTED_IMG_SIZE = (162, 512)
branch2_dim = [2, 32, 32, 64]  
trunk_dim = [2, 100, 100, 64]  
geometry_dim = EXPECTED_IMG_SIZE

# base paths
data_folder   = r"data/"
model_folder  = r"result/"

# dataset subfolders
DATA_PATH_IMAGES      = os.path.join(data_folder, "images")
DATA_PATH_SIMULATIONS = os.path.join(data_folder, "simulation_outputs")

# checkpoint + results folders
CHECKPOINT_PATH = os.path.join(model_folder, "model_checkpoint.pth")
RESULTS_FOLDER  = os.path.join(model_folder, "inferences1")

os.makedirs(RESULTS_FOLDER, exist_ok=True)
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "deeponet" in model_type:
    model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

# prepare dataset
image_paths, _ = get_paths(DATA_PATH_IMAGES)  
_, simulation_paths = get_paths(DATA_PATH_SIMULATIONS)  
print(f"Found {len(image_paths)} image files.")
print(f"Found {len(simulation_paths)} simulation files.")

if len(image_paths) == 0 or len(simulation_paths) == 0:
    raise ValueError("No image or simulation files found. Please check the dataset structure or file extensions.")

# test dataset loader
test_dataset = TransducerDataset(image_paths, simulation_paths, loading_method='individual', device=device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

if model_type == "geofno":
    # --- peek one batch to infer source_dim (matches training) ---
    try:
        image0, transducer_locs0, locs0, _ = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("Empty dataset for inference.")

    source_dim = transducer_locs0.shape[-1]

    # --- construct GeoFNO with the SAME hparams used in training ---
    model = GeoFNO(
        geom_channels=3,
        source_dim=source_dim,
        fno_width=64,
        fno_layers=4,
        modes1=16,
        modes2=16,
        geom_feat_ch=32,
        source_feat_ch=32,
        posenc_frequencies=4,
        use_deformation=True,
        mask_loss=False
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = (
        ckpt.get("model_state_dict", None)
        or ckpt.get("state_dict", None)
        or (ckpt if isinstance(ckpt, dict) and all(k.startswith(("geom_enc","src_enc","fno","out")) for k in ckpt.keys()) else None)
    )
    if state_dict is None:
        raise ValueError(f"Unrecognized checkpoint format: keys={list(ckpt.keys())[:10]}")
    model.load_state_dict(state_dict)
    model.eval()




visualizer = SegmentationVisualizer()
losses = []
img_loss_log = "inference_image_loss_log.txt"
for i, (image, transducer_locs, locs, simulations) in enumerate(test_loader):
    print(f"Processing image {i+1}")
    
    image = image.to(device)
    transducer_locs = transducer_locs.to(device)
    locs = locs.to(device)
    simulations = simulations.to(device)
    
    # for timing inference 
    #start_time = time.time()
    with torch.no_grad():
        prediction = model(image, transducer_locs, locs)
    #end_time = time.time()

    # print(f"Time taken for inference on image {i+1}: {end_time - start_time:.6f} seconds")

    #print(f"Prediction shape for image {i+1}: {prediction.shape}")
    images01 = visualizer.minmax_normalize(image)
    simulations01 = visualizer.minmax_normalize(simulations)
    prediction01 = visualizer.minmax_normalize(prediction)

    comment = f'inference_image_{i+1}'
    visualizer.visualize_batch(images01, simulations01, prediction01, batch=1, comment=comment, result_folder=RESULTS_FOLDER)
    print(f"Saved prediction {i+1}")

    # Calculate loss
    numerator = torch.norm(prediction - simulations, p=2)
    denominator = torch.norm(simulations, p=2)  # Avoid division by zero
    loss = (numerator / denominator) ** 2
    losses.append(loss.item())
    with open(os.path.join(RESULTS_FOLDER, img_loss_log), 'a') as f:
        f.write(f"Image {i+1}: Loss = {loss.item():.6f}\n")

print(f"Average loss over {len(test_loader)} images: {np.mean(losses):.6f}")