# geofno_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Small utilities
# ---------------------------

def _posenc(coords, num_frequencies=4):
    """
    Sinusoidal positional encoding for coordinates in [-1, 1].
    coords: (B, H, W, 2)
    returns: (B, 2 * 2 * num_frequencies, H, W)
             (sin+cos for x and y, across K frequencies)
    """
    B, H, W, C = coords.shape
    assert C == 2
    # (B, H, W, 1) each
    x = coords[..., 0:1]
    y = coords[..., 1:2]

    # Frequencies: 1, 2, 4, 8, ... up to num_frequencies
    freqs = torch.tensor([2.0 ** k for k in range(num_frequencies)],
                         device=coords.device, dtype=coords.dtype).view(1, 1, 1, num_frequencies)

    # shape -> (B, H, W, K)
    xw = math.pi * x * freqs
    yw = math.pi * y * freqs

    sinx = torch.sin(xw)
    cosx = torch.cos(xw)
    siny = torch.sin(yw)
    cosy = torch.cos(yw)

    # concat over last dim -> (B,H,W,4K) then permute to NCHW
    feats = torch.cat([sinx, cosx, siny, cosy], dim=-1)
    feats = feats.permute(0, 3, 1, 2).contiguous()
    return feats


# ---------------------------
# Spectral Convolution (FNO-2D core)
# ---------------------------

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Learned complex weights on a limited number of low-frequency modes.
    """
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex weights (real/imag as separate Parameters)
        scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def compl_mul2d(self, input_ft, weight_real, weight_imag):
        # input_ft: (B, C_in, H_ft, W_ft) complex tensor (torch.complex64/128)
        # Fourier domain
        B, C_in, H_ft, W_ft = input_ft.shape
        m1 = min(self.modes1, H_ft)
        m2 = min(self.modes2, W_ft)

        out_ft = input_ft.new_zeros(B, self.out_channels, H_ft, W_ft)

        # (B, C_in, m1, m2) x (C_in, C_out, m1, m2) -> (B, C_out, m1, m2)
        in_slice = input_ft[:, :C_in, :m1, :m2]
        wr = weight_real[:C_in, :self.out_channels, :m1, :m2]
        wi = weight_imag[:C_in, :self.out_channels, :m1, :m2]

        # Manual complex multiply and sum over C_in
        # (B, C_out, m1, m2)
        out_real = torch.einsum('bcxy, coxy -> boxy', in_slice.real, wr) - torch.einsum('bcxy, coxy -> boxy', in_slice.imag, wi)
        out_imag = torch.einsum('bcxy, coxy -> boxy', in_slice.real, wi) + torch.einsum('bcxy, coxy -> boxy', in_slice.imag, wr)

        out_ft[:, :, :m1, :m2] = torch.complex(out_real, out_imag)
        return out_ft

    def forward(self, x):
        """
        x: (B, C_in, H, W) -> (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        # rfft2 returns (B, C, H, W//2+1) in complex
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = self.compl_mul2d(x_ft, self.weight_real, self.weight_imag)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNOBlock2D(nn.Module):
    """
    A stack: SpectralConv2d + 1x1 Conv (local mixing) + GELU
    """
    def __init__(self, width, modes1=16, modes2=16):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.spectral(x)
        x2 = self.local(x)
        return self.act(x1 + x2)


# ---------------------------
# Deformation network (Geo part)
# ---------------------------

class DeformationNet(nn.Module):
    """
    Maps physical coords (and optional global conditioning) -> latent coords.
    If disabled, we pass coords through (identity).
    """
    def __init__(self, in_dim=2, cond_dim=0, hidden=64, depth=3):
        super().__init__()
        dims = [in_dim + cond_dim] + [hidden] * (depth - 1) + [2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, coords, cond=None):
        """
        coords: (B, H, W, 2)
        cond:   (B, cond_dim) or None
        returns latent_coords in [-1,1]: (B, H, W, 2)
        """
        B, H, W, _ = coords.shape
        if cond is not None:
            cond_expanded = cond.view(B, 1, 1, -1).expand(B, H, W, -1)
            x = torch.cat([coords, cond_expanded], dim=-1)
        else:
            x = coords
        out = self.net(x)  # (B, H, W, 2)
        # optional tanh to keep latent grid within [-1,1]
        return torch.tanh(out)


# ---------------------------
# Geometry & source encoders
# ---------------------------

class GeometryEncoder(nn.Module):
    """
    Simple CNN -> per-pixel features aligned to (H, W)
    """
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, geometry):
        x = self.act(self.conv1(geometry))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x  # (B, out_ch, H, W)


class SourceEncoder(nn.Module):
    """
    MLP -> global conditioning vector, broadcast later
    """
    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.ReLU(inplace=True)
        )

    def forward(self, s):
        return self.net(s)  # (B, out_dim)


# ---------------------------
# Geo-FNO Model
# ---------------------------

class GeoFNO(nn.Module):
    """
    Inputs:
      geometry:   (B, 3, H, W)
      source_loc: (B, S)
      coords:     (B, H, W, 2)   # normalized to [-1, 1]

    Output:
      y_pred:     (B, H, W)      # scalar field

    Outline:
      1) Encode geometry -> per-pixel features G (B, Cg, H, W)
      2) Encode source -> global cond vector S (B, Cs), broadcast to (B, Cs, H, W)
      3) Optional deformation: coords -> latent_coords; resample G via grid_sample onto latent grid
      4) Positional encodings from (latent_)coords
      5) Concatenate [posenc, G (resampled), S (broadcast)] -> FNO stack
      6) Head -> scalar field
    """
    def __init__(
        self,
        geom_channels=3,
        source_dim=2,
        fno_width=64,
        fno_layers=4,
        modes1=16,
        modes2=16,
        geom_feat_ch=32,
        source_feat_ch=32,
        posenc_frequencies=4,
        use_deformation=True,
        mask_loss=True
    ):
        super().__init__()
        self.use_deformation = use_deformation
        self.mask_loss = mask_loss

        # Encoders
        self.geom_enc = GeometryEncoder(in_ch=geom_channels, out_ch=geom_feat_ch)
        self.src_enc = SourceEncoder(in_dim=source_dim, out_dim=source_feat_ch)

        # Deformation: coords (+ optional source cond) -> latent coords
        self.deform = DeformationNet(in_dim=2, cond_dim=source_feat_ch if use_deformation else 0,
                                     hidden=64, depth=3)

        # First lifting (1x1 conv) after concatenation
        pe_ch = 4 * posenc_frequencies  # sin/cos for x & y -> 4K channels
        in_ch = geom_feat_ch + source_feat_ch + pe_ch
        self.lift = nn.Conv2d(in_ch, fno_width, kernel_size=1)

        # FNO stack
        self.fno_layers = nn.ModuleList(
            [FNOBlock2D(fno_width, modes1=modes1, modes2=modes2) for _ in range(fno_layers)]
        )

        # Projection head
        self.proj = nn.Sequential(
            nn.Conv2d(fno_width, fno_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fno_width, 1, kernel_size=1)
        )

        self.posenc_frequencies = posenc_frequencies

    def forward(self, geometry, source_loc, coords, mask=None):
        """
        geometry:   (B, 3, H, W)
        source_loc: (B, S)
        coords:     (B, H, W, 2) in [-1,1]
        mask:       (B, 1, H, W) or (B, H, W) optional
        """
        B, _, H, W = geometry.shape

        # Encoders
        G = self.geom_enc(geometry)               # (B, Cg, H, W)
        Sg = self.src_enc(source_loc)             # (B, Cs)
        S = Sg.view(B, -1, 1, 1).expand(B, -1, H, W)  # (B, Cs, H, W)

        # Deform coords (and optionally resample geometry features to the latent grid)
        if self.use_deformation:
            latent_coords = self.deform(coords, Sg)  # (B, H, W, 2) in [-1,1]
            # grid_sample expects NCHW input and grid (N,H,W,2) in [-1,1]
            G_latent = F.grid_sample(G, grid=latent_coords, mode="bilinear",
                                     align_corners=True)
            PE = _posenc(latent_coords, num_frequencies=self.posenc_frequencies)
            x_in = torch.cat([G_latent, S, PE], dim=1)
        else:
            PE = _posenc(coords, num_frequencies=self.posenc_frequencies)
            x_in = torch.cat([G, S, PE], dim=1)

        # Lift -> FNO stack -> head
        x = self.lift(x_in)
        for blk in self.fno_layers:
            x = blk(x)
        y = self.proj(x).squeeze(1)  # (B, H, W)
        return y

    @staticmethod
    def relative_l2(pred, target, mask=None, eps=1e-12):
        if mask is not None:
            # accept (B,1,H,W) or (B,H,W)
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.squeeze(1)
            pred = pred * mask
            target = target * mask
        num = torch.norm(pred - target, p=2)
        den = torch.norm(target, p=2).clamp_min(eps)
        return (num / den) ** 2

    def loss(self, geometry, source_loc, coords, target_field, mask=None):
        pred = self.forward(geometry, source_loc, coords, mask=mask)
        return self.relative_l2(pred, target_field, mask=mask)


# ---------------------------
# Example usage / training stub
# ---------------------------

if __name__ == "__main__":
    # match these to the real tensors
    B, H, W = 4, 128, 192
    geom = torch.randn(B, 3, H, W)            # geometry image(s)
    source_loc = torch.randn(B, 4)            # e.g., (x_s, y_s, intensity, ...)
    coords = torch.rand(B, H, W, 2) * 2 - 1   # normalize coords to [-1, 1]
    target = torch.randn(B, H, W)             # ground-truth scalar field
    mask = torch.ones(B, 1, H, W)             # optional: 1 inside domain, 0 outside

    model = GeoFNO(
        geom_channels=3,
        source_dim=source_loc.shape[1],
        fno_width=64,
        fno_layers=4,
        modes1=16,
        modes2=16,
        geom_feat_ch=32,
        source_feat_ch=32,
        posenc_frequencies=4,
        use_deformation=True,
        mask_loss=True
    )

    # Quick forward + loss
    pred = model(geom, source_loc, coords, mask=mask)
    loss = model.loss(geom, source_loc, coords, target, mask=mask)
    print("Pred shape:", pred.shape, "Loss:", loss.item())

    # Minimal training step (mirror your current training loop)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
