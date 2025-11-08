# Training script for Geo-FNO
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

from dataset_prep import get_paths, TransducerDataset
from utils import (
    log_loss, plot_logs, plot_prediction, store_model,
    ensure_directory_exists, get_time_YYYYMMDDHH
)

from geo_fno import GeoFNO


# 1) Paths
with open('config.json', 'r') as file:
    config = json.load(file)
DATA_PATH = config['DATA_PATH']
RESULT_FOLDER = config['RESULT_FOLDER']
if config['win']:
    DATA_PATH = rf'{DATA_PATH}'
    RESULT_FOLDER = rf'{RESULT_FOLDER}'

hyp_params = {'batch': config['BATCHSIZE'],
               'lr': config['lr'], 
               'epochs': config['epochs'],
                  }

################ NNI ###########################

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

##################### NNI ########################

# 2) Device control
CPU_ONLY        = False               # keep True to force CPU-only
DEVICE_FALLBACK =  "cuda"     # used only if CPU_ONLY=False and CUDA is available

# 3) Data shape
EXPECTED_IMG_SIZE = (162, 512)
EXPECTED_SIM_SIZE = (162, 512)

# 4) Training hyperparameters (essentials only)
EPOCHS          = hyp_params['epochs']
BATCHSIZE       = hyp_params['batch']
LR              = hyp_params['lr']
WEIGHT_DECAY    = 1e-6
PLATEAU_PATIENCE= 10
PLATEAU_FACTOR  = 0.5

# 5) Geo-FNO hyperparameters (the important knobs)
FNO_WIDTH       = 64
FNO_LAYERS      = 4
MODES1          = 16
MODES2          = 16
GEOM_FEAT_CH    = 32
SOURCE_FEAT_CH  = 32
POSENC_K        = 4
USE_DEFORMATION = True   

# 6) Visualization cadence
VIZ_EVERY       = 10      # epochs; also runs at the final epoch


# =========================
# Data loading 
# =========================
def load_data_by_split(data_path, bz, shuffle=True):
    print('-'*12, 'DATA READIN BY SPLIT', '-'*12)
    split_path_dict = {}
    for split_name in ['train', 'val', 'test']:
        split_data_path = os.path.join(data_path, '{data_type}', split_name)
        images_path, simulation_path = get_paths(split_data_path)
        dataset_ = TransducerDataset(images_path, simulation_path, loading_method='individual')
        dataloader_ = DataLoader(dataset_, batch_size=bz, shuffle=shuffle, num_workers=0)
        split_path_dict[split_name] = dataloader_
    return list(split_path_dict.values())  # train, val, test


# =========================
# Trainer
# =========================
class Trainer:
    def __init__(self, model, optimizer, device, num_epochs=100):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def set_result_path(self, result_path):
        self.result_folder = result_path
        self.val_log_path = os.path.join(self.result_folder, 'loss_log_val.txt')
        self.train_log_path = os.path.join(self.result_folder, 'loss_log_train.txt')
        print(f"[INFO] Results will be stored at: {result_path}")

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        for geometry, source_loc, coords, labels in dataloader:
            geometry   = geometry.to(self.device)     # (B,3,H,W)
            source_loc = source_loc.to(self.device)   # (B,S)
            coords     = coords.to(self.device)       # (B,H,W,2) in [-1,1]
            labels     = labels.to(self.device)       # (B,H,W)

            loss = self.model.loss(geometry, source_loc, coords, labels, mask=None)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        return avg_loss

    def val_one_epoch(self, dataloader_validation):
        self.model.eval()
        total_val_loss, num_batches = 0.0, 0
        with torch.no_grad():
            for geometry, source_loc, coords, labels in dataloader_validation:
                geometry   = geometry.to(self.device)
                source_loc = source_loc.to(self.device)
                coords     = coords.to(self.device)
                labels     = labels.to(self.device)

                val_loss = self.model.loss(geometry, source_loc, coords, labels, mask=None)
                total_val_loss += val_loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def test(self, dataloader_test):
        self.model.eval()
        total_test_loss, num_batches = 0.0, 0
        with torch.no_grad():
            for batch, (geometry, source_loc, coords, labels) in enumerate(dataloader_test):
                geometry   = geometry.to(self.device)
                source_loc = source_loc.to(self.device)
                coords     = coords.to(self.device)
                labels     = labels.to(self.device)

                test_loss = self.model.loss(geometry, source_loc, coords, labels, mask=None)
                total_test_loss += test_loss.item()
                num_batches += 1

                pred = self.model(geometry, source_loc, coords)
                plot_prediction(geometry.cpu(), labels.cpu(), pred.cpu(), batch,
                                result_folder=self.result_folder)
        avg_test_loss = total_test_loss / max(1, num_batches)
        self.test_loss = avg_test_loss
        return avg_test_loss

    def visualize_prediction_subset(self, dataloader, comment='val', subset_size=4):
        # Visualize a small subset for speed
        self.model.eval()
        with torch.no_grad():
            subset_indices = list(range(min(subset_size, len(dataloader.dataset))))
            subset_dataset = Subset(dataloader.dataset, subset_indices)
            subset_loader  = DataLoader(subset_dataset, batch_size=subset_size, shuffle=False)
            for batch, (geometry, source_loc, coords, labels) in enumerate(subset_loader):
                pred = self.model(geometry.to(self.device), source_loc.to(self.device), coords.to(self.device))
                plot_prediction(geometry.cpu(), labels.cpu(), pred.cpu(), batch,
                                comment=comment, result_folder=self.result_folder)
                break

    def visualize_deformation_grid(self, dataloader, comment='deform', subsample=8):
        # Optional: show latent coords x', y' if deformation is enabled
        if not getattr(self.model, "use_deformation", False):
            return
        import matplotlib.pyplot as plt
        self.model.eval()
        with torch.no_grad():
            (geom, src, coords, labels) = next(iter(dataloader))
            geom   = geom.to(self.device)
            src    = src.to(self.device)
            coords = coords.to(self.device)

            G  = self.model.geom_enc(geom)
            Sg = self.model.src_enc(src)
            lat = self.model.deform(coords, Sg)[0]  # (H,W,2)

            H, W, _ = lat.shape
            xs = lat[::max(1,H//subsample), ::max(1,W//subsample), 0].cpu().numpy()
            ys = lat[::max(1,H//subsample), ::max(1,W//subsample), 1].cpu().numpy()

            import numpy as np
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            im0 = axes[0].imshow(xs, origin='lower'); axes[0].set_title("latent x'")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(ys, origin='lower'); axes[1].set_title("latent y'")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            for ax in axes: ax.axis('off')
            out_path = os.path.join(self.result_folder, f"latent_coords_{comment}.png")
            fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

    def train(self, train_loader, val_loader, test_loader, scheduler,
              viz_every=10, do_deform_viz=True):
        ensure_directory_exists(self.result_folder)

        for epoch in range(self.num_epochs):
            tr = self.train_one_epoch(train_loader)
            va = self.val_one_epoch(val_loader)
            scheduler.step(va)

            log_loss(tr, temp_file=self.train_log_path)
            log_loss(va, self.val_log_path)

            print(f"[Epoch {epoch+1:03d}/{self.num_epochs}]  Train {tr:.6f} | Val {va:.6f}")

            if (epoch + 1) % viz_every == 0 or (epoch + 1) == self.num_epochs:
                self.visualize_prediction_subset(val_loader, comment=f'val_ep{epoch+1}', subset_size=4)
                if do_deform_viz:
                    self.visualize_deformation_grid(val_loader, comment=f'ep{epoch+1}')

        if test_loader is not None:
            te = self.test(test_loader)
            print(f"[Test] Loss: {te:.6f}")

        store_model(self.model, self.optimizer, self.num_epochs, self.result_folder)
        return self.model


# =========================
# Build & run
# =========================
def build_model(source_dim: int):
    from geo_fno import GeoFNO  
    return GeoFNO(
        geom_channels=3,
        source_dim=source_dim,       
        fno_width=FNO_WIDTH,
        fno_layers=FNO_LAYERS,
        modes1=MODES1,
        modes2=MODES2,
        geom_feat_ch=GEOM_FEAT_CH,
        source_feat_ch=SOURCE_FEAT_CH,
        posenc_frequencies=POSENC_K,
        use_deformation=USE_DEFORMATION,
        mask_loss=False
    )

def select_device():
    if CPU_ONLY:
        return torch.device("cpu")
    if DEVICE_FALLBACK == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if DEVICE_FALLBACK == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    device = select_device()
    print(f"[CONFIG] DATA_PATH={DATA_PATH}  RESULT_FOLDER={RESULT_FOLDER}  EPOCHS={EPOCHS}  "
          f"BATCH={BATCHSIZE}  IMG={EXPECTED_IMG_SIZE}  SIM={EXPECTED_SIM_SIZE}  DEVICE={device}")

    # result dir with timestamp
    run_dir = os.path.join(RESULT_FOLDER, get_time_YYYYMMDDHH() + "_geofno")
    ensure_directory_exists(run_dir)

    # -------------------------------
    # 1 Data
    # -------------------------------
    train_loader, val_loader, test_loader = load_data_by_split(DATA_PATH, BATCHSIZE, shuffle=True)

    # Peek one batch to get source_dim (length of source_loc vector)
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Training dataloader is empty. Check your DATA_PATH and file structure.")
    geometry, source_loc, coords, labels = sample_batch
    inferred_source_dim = source_loc.shape[-1]
    print(f"[INFO] Inferred source_dim from data: {inferred_source_dim}")

    # --------------------------------------
    # 2) Build model
    # --------------------------------------
    model = build_model(inferred_source_dim).to(device)
    print(f"[MODEL] Params total: {sum(p.numel() for p in model.parameters()):,} | "
          f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer & scheduler
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=PLATEAU_FACTOR,
                                  patience=PLATEAU_PATIENCE, verbose=True)

    # Train
    trainer = Trainer(model, optimizer, device, num_epochs=EPOCHS)
    trainer.set_result_path(run_dir)
    trainer.train(train_loader, val_loader, test_loader, scheduler,
                  viz_every=VIZ_EVERY, do_deform_viz=True)

    # Loss plots
    plot_logs([trainer.train_log_path, trainer.val_log_path],
              output_image=os.path.join(trainer.result_folder, "loss_plot.png"))

if __name__ == "__main__":
    main()
