"""
Unified Trainer class for all model types (CNN, ViT, FNO)
Supports different model APIs through flexible input handling
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from utils import log_loss, plot_prediction, store_model, ensure_directory_exists
from dataset_prep import get_paths, TransducerDataset


class Trainer:
    """
    Unified trainer that supports both OPNN (CNN/ViT) and GeoFNO models.
    
    Model API conventions:
    - OPNN models: loss(branch1, branch2, trunk, labels), forward returns (pred, _)
    - GeoFNO models: loss(geometry, source_loc, coords, labels, mask=None), forward returns pred
    """
    
    def __init__(self, model, optimizer, device, num_epochs=1000, model_type='opnn'):
        """
        Args:
            model: The model to train
            optimizer: Optimizer instance
            device: torch device
            num_epochs: Number of training epochs
            model_type: 'opnn' for CNN/ViT models, 'fno' for GeoFNO models
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def set_result_path(self, result_path):
        self.result_folder = result_path
        self.val_log_path = os.path.join(self.result_folder, 'loss_log_val.txt')
        self.train_log_path = os.path.join(self.result_folder, 'loss_log_train.txt')
        print(f"Result of this Training will be stored at {result_path}")

    def _compute_loss(self, *inputs):
        """Unified loss computation that handles both model types"""
        if self.model_type == 'opnn':
            # OPNN: loss(branch1, branch2, trunk, labels)
            branch1_input, branch2_input, trunk_input, labels = inputs
            return self.model.loss(branch1_input, branch2_input, trunk_input, labels)
        else: 
            geometry, source_loc, coords, labels = inputs
            return self.model.loss(geometry, source_loc, coords, labels, mask=None)

    def _forward(self, *inputs):
        """Unified forward pass that handles both model types"""
        if self.model_type == 'opnn':

            branch1_input, branch2_input, trunk_input = inputs[:3]
            prediction= self.model(branch1_input, branch2_input, trunk_input)
            return prediction
        else:
            geometry, source_loc, coords = inputs[:3]
            return self.model(geometry, source_loc, coords)

    def train_one_epoch(self, dataloader, clip_grad_norm=None):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            # Move all inputs to device
            inputs = [item.to(self.device) for item in batch_data[:-1]]
            labels = batch_data[-1].to(self.device)
            
            # Compute loss
            loss = self._compute_loss(*inputs, labels)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Optional gradient clipping (for FNO)
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        return avg_loss

    def val_one_epoch(self, dataloader_validation):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader_validation:
                inputs = [item.to(self.device) for item in batch_data[:-1]]
                labels = batch_data[-1].to(self.device)
                
                val_loss = self._compute_loss(*inputs, labels)
                total_val_loss += val_loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / max(1, num_batches)
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def test(self, dataloader_test):
        self.model.eval()
        total_test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch, batch_data in enumerate(dataloader_test):
                inputs = [item.to(self.device) for item in batch_data[:-1]]
                labels = batch_data[-1].to(self.device)
                
                test_loss = self._compute_loss(*inputs, labels)
                total_test_loss += test_loss.item()
                num_batches += 1
                
                # Plot sample prediction
                prediction = self._forward(*inputs)
                # Get the first input (geometry/branch1) for visualization
                first_input = batch_data[0].cpu()
                plot_prediction(first_input, labels.cpu(), prediction.cpu(), batch, 
                              result_folder=self.result_folder)
        
        avg_test_loss = total_test_loss / max(1, num_batches)
        self.test_loss = avg_test_loss
        return avg_test_loss
    
    def visualize_prediction(self, dataloader, comment='', subset=True, batch_size=None):
        """Visualize predictions (compatible with both OPNN and FNO)"""
        if subset:
            subset_size = batch_size if batch_size is not None else 4
            subset_indices = list(range(min(subset_size, len(dataloader.dataset))))
            subset_dataset = Subset(dataloader.dataset, subset_indices)
            dataloader = DataLoader(subset_dataset, batch_size=subset_size, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            for batch, batch_data in enumerate(dataloader):
                inputs = [item.to(self.device) for item in batch_data[:-1]]
                labels = batch_data[-1]
                
                prediction = self._forward(*inputs)
                first_input = batch_data[0].cpu()
                plot_prediction(first_input, labels.cpu(), prediction.cpu(), batch,
                              comment=comment, result_folder=self.result_folder)
                if subset:
                    break  # Only visualize one batch for subset
        return True

    def visualize_deformation_grid(self, dataloader, comment='deform', subsample=8):
        """Optional: show latent coords if deformation is enabled (FNO only)"""
        if self.model_type != 'fno' or not getattr(self.model, "use_deformation", False):
            return
        import matplotlib.pyplot as plt
        import numpy as np
        
        self.model.eval()
        with torch.no_grad():
            batch_data = next(iter(dataloader))
            geometry = batch_data[0].to(self.device)
            source_loc = batch_data[1].to(self.device)
            coords = batch_data[2].to(self.device)
            
            # Calculate latent coords like forward()
            G = self.model.geom_enc(geometry)
            Sg = self.model.src_enc(source_loc)
            lat = self.model.deform(coords, Sg)[0]  # (H,W,2)
            
            H, W, _ = lat.shape
            xs = lat[::max(1, H//subsample), ::max(1, W//subsample), 0].cpu().numpy()
            ys = lat[::max(1, H//subsample), ::max(1, W//subsample), 1].cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            im0 = axes[0].imshow(xs, origin='lower')
            axes[0].set_title("latent x'")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(ys, origin='lower')
            axes[1].set_title("latent y'")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            for ax in axes:
                ax.axis('off')
            out_path = os.path.join(self.result_folder, f"latent_coords_{comment}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    def train(self, dataloader, dataloader_validation, dataloader_test, scheduler,
              viz_every=None, do_deform_viz=False, clip_grad_norm=None):
        """
        Unified training loop
        
        Args:
            dataloader: Training dataloader
            dataloader_validation: Validation dataloader
            dataloader_test: Test dataloader (can be None)
            scheduler: Learning rate scheduler
            viz_every: Visualize every N epochs (None to disable periodic visualization)
            do_deform_viz: Whether to visualize deformation grid (FNO only)
            clip_grad_norm: Gradient clipping max norm (None to disable)
        """
        ensure_directory_exists(self.result_folder)
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(dataloader, clip_grad_norm=clip_grad_norm)
            val_loss = self.val_one_epoch(dataloader_validation)
            scheduler.step(val_loss)
            
            log_loss(train_loss, temp_file=self.train_log_path)
            log_loss(val_loss, self.val_log_path)
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            
            # Periodic visualization (if enabled)
            if viz_every is not None and (epoch + 1) % viz_every == 0:
                self.visualize_prediction(dataloader_validation, comment=f'val_ep{epoch+1}', subset=True)
                if do_deform_viz:
                    self.visualize_deformation_grid(dataloader_validation, comment=f'ep{epoch+1}')
        
        # Final visualization
        if viz_every is not None:
            self.visualize_prediction(dataloader_validation, comment=f'val_ep{self.num_epochs}', subset=True)
            if do_deform_viz:
                self.visualize_deformation_grid(dataloader_validation, comment=f'ep{self.num_epochs}')
        
        # Test Model
        if dataloader_test is not None:
            test_loss = self.test(dataloader_test)
            print(f"Test Loss: {test_loss:.4f}")
        
        # Store model
        store_model(self.model, self.optimizer, epoch, self.result_folder)
        return self.model


def load_data_by_split(data_path, bz, shuffle=True):
    """
    Load data split into train, validation, and test dataloaders.
    
    Args:
        data_path: Base path containing {data_type}/{train,val,test}
        bz: Batch size
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print('-'*15, 'DATA READIN BY SPLIT', '-'*15)
    split_path_dict = {}
    for split_name in ['train', 'val', 'test']:
        split_data_path = os.path.join(data_path, '{data_type}', split_name)
        images_path, simulation_path = get_paths(split_data_path)
        dataset_ = TransducerDataset(images_path, simulation_path, loading_method='individual')
        dataloader_ = DataLoader(dataset_, batch_size=bz, shuffle=shuffle, num_workers=0)
        split_path_dict[split_name] = dataloader_
    
    return list(split_path_dict.values())  # train, val, test

