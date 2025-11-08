import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#deep_env\Scripts\activate

class opnn_transformer(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, geometry_dim, model_type = '162_training_vitcustom'):
        super(opnn_transformer, self).__init__()
        """
        model_type: <input_dim>_<pretrain/training requirement>_<model>
        - input_dim: 
        --- 162 for original ultrasound image
        --- 224 for pretrained ViT (change config "image_transforms":"original" -> "transformer" to crop in dataset_prep.py)
        - pretrain/training requirement:
        --- pretrained: use pretrained ViT model (requires 224 input), freeze all parameters but last fc
        --- finetuned: pretrain Vit middle layer (freeze) + customize Input / Output layer to match in/out size
        --- training: train all parameters
        - model: 
        --- vit: use ViT model (requires 224 input)
        --- vitcustom: customize ViT model
        """
        input_dim, train_status, model_name = model_type.split('_')

        if model_name == 'vitcustom':
            self.vit= VisionTransformerCustom(img_size=(162, 512), in_channels=3)
        else:

            ...

        # Fully connected layer to adjust ViT output size
        self.fc1 = nn.Linear(768, 64)

        # Source location branch 
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], branch2_dim[3])
        )

        # Trunk network
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], branch2_dim[3])
        )

    def forward(self, geometry, source_loc, coords):
        # Process geometry image through ViT (frozen)
        # with torch.no_grad():
        #     x = self.vit(geometry)  # Shape: (batch, 768)
        x = self.vit(geometry)  # Shape: (batch, 64)

        y_br1 = F.relu(self.fc1(x))  # Shape: (batch, 64)

        # Process source location through FC network
        y_br2 = self._branch2(source_loc)  # Shape: (batch, 64)

        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)

        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("bf,bhwf->bhw", y_br, y_tr)

        return y_out

    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        numerator = torch.norm(y_out - target_pressure, p=2)
        denominator = torch.norm(target_pressure, p=2)  # Avoid division by zero
        loss = (numerator / denominator) ** 2
        return loss



class VisionTransformerCustom(nn.Module):
    def __init__(self, img_size=(162, 512), patch_size=16, in_channels=3, emb_dim=128, num_heads=4, num_layers=4, output_dim=768):
        super(VisionTransformerCustom, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # Number of patches
        self.patch_dim = in_channels * patch_size * patch_size  # Flattened patch dimension

        # Patch embedding layer
        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output mapping to 64 dimensions
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, output_dim)
        )
        # self.freeze_middle_layers()

    def freeze_middle_layers(self):
        """Freezes the encoder while keeping the first and last layers trainable."""
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the transformer encoder

        for param in self.patch_embed.parameters():
            param.requires_grad = True  # Allow training of patch embedding layer

        for param in self.mlp_head.parameters():
            param.requires_grad = True  # Allow training of the final fully connected layer


    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Convert image into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.num_patches, -1)  # (batch, num_patches, patch_dim)

        # Apply linear patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, emb_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_patches + 1, emb_dim)

        # Add positional encoding
        x = x + self.pos_embed

        # Pass through transformer encoder
        x = self.encoder(x)
        
        # Use class token for final representation
        x = x[:, 0]  # Take only the class token
        x = self.mlp_head(x)  # Map to 64 dimensions

        return x

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

# Load a pretrained ViT model
pretrained_model = vit_b_16(weights="IMAGENET1K_V1")  # Load ImageNet-pretrained ViT

class VisionTransformerCustom_pretrained(nn.Module):
    def __init__(self, output_dim=768):
        super(VisionTransformerCustom_pretrained, self).__init__()

        # Use the pretrained patch embedding and transformer encoder
        self.patch_embed = pretrained_model.conv_proj  # Patch embedding layer
        self.encoder = pretrained_model.encoder  # Transformer encoder

        # Freeze the transformer encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # Prevent updates during training

        # Define a new MLP head for fine-tuning
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, output_dim)
        )

    def forward(self, x):
        # Apply patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # Convert image to patch embeddings

        # Add class token
        cls_token = pretrained_model.class_token.expand(x.shape[0], -1, -1).to('cuda')
        x = torch.cat((cls_token, x), dim=1)  # Concatenate class token

        # Add positional embedding
        x = x + pretrained_model.encoder.pos_embedding

        # Pass through frozen transformer encoder
        with torch.no_grad():  # No gradient updates for encoder
            x = self.encoder(x)

        # Use the class token output
        x = x[:, 0]  # Extract class token
        x = self.mlp_head(x)  # Final output mapping

        return x



