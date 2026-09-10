"""
Load GRU-VAE checkpoint and export to ONNX.
Uses a custom unpickler that properly handles PyTorch storage references.
"""
import os
import json
import pickle
import io
import numpy as np
import torch
import torch.nn as nn
import torch.onnx

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'best_config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

N_FEATURES = 6
WINDOW_SIZE = config.get("window_size", 20)
HIDDEN_DIM = config.get("hidden_dim", 16)
LATENT_DIM = config.get("latent_dim", 8)


class GRUVAE(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, window_size=WINDOW_SIZE):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_gru = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, h = self.encoder_gru(x)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h0 = self.decoder_input(z).unsqueeze(0)
        dec_in = h0.transpose(0, 1).repeat(1, self.window_size, 1)
        out, _ = self.decoder_gru(dec_in, h0)
        return self.output_layer(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class StorageMapper:
    """Maps storage IDs to tensor data."""
    def __init__(self, storage_dir):
        self.storages = {}
        self._load_storages(storage_dir)
    
    def _load_storages(self, storage_dir):
        for fname in os.listdir(storage_dir):
            if fname.isdigit():
                fpath = os.path.join(storage_dir, fname)
                with open(fpath, 'rb') as f:
                    # Try to interpret as different dtypes
                    data_f32 = np.frombuffer(f.read(), dtype=np.float32)
                    self.storages[int(fname)] = data_f32
    
    def get_storage(self, storage_id):
        if storage_id not in self.storages:
            raise ValueError(f"Storage ID {storage_id} not found")
        return self.storages[storage_id]


class CustomUnpickler(pickle.Unpickler):
    def __init__(self, file, storage_mapper):
        super().__init__(file)
        self.storage_mapper = storage_mapper
    
    def persistent_load(self, pid):
        # Format: ('storage', <class>, 'id', 'device', size)
        if isinstance(pid, tuple) and len(pid) >= 5:
            prefix, storage_type, storage_id, device, size = pid
            if prefix == 'storage':
                raw_data = self.storage_mapper.get_storage(int(storage_id))
                # Create a simple storage-like object
                class SimpleStorage:
                    def __init__(self, data):
                        self._data = data
                        self.dtype = np.float32
                        self.device = torch.device('cpu')
                    def __len__(self):
                        return len(self._data)
                    def cast(self, dtype, device=None):
                        return self._data.astype(np.float32)
                return SimpleStorage(raw_data)
        raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    """Rebuild a tensor from storage."""
    import torch
    # Create tensor from storage
    numel = storage._data[storage_offset:storage_offset + size[0] * size[1] if len(size) > 1 else storage_offset + size[0]].size if hasattr(storage, '_data') else 0
    # Simpler approach: just return the raw data reshaped
    return storage._data


def main():
    base_dir = os.path.dirname(__file__)
    storage_dir = os.path.join(base_dir, 'gru_vae_best', 'data')
    pickle_path = os.path.join(base_dir, 'gru_vae_best', 'data.pkl')
    
    print(f"Loading storage from {storage_dir}...")
    storage_mapper = StorageMapper(storage_dir)
    
    print("Loading state dict from pickle...")
    with open(pickle_path, 'rb') as f:
        # Use standard torch unpickler but with our storage mapper
        unpickler = CustomUnpickler(f, storage_mapper)
        
        # Monkey-patch _rebuild_tensor_v2 to use our storage
        original_rebuild = torch._utils._rebuild_tensor_v2
        
        def custom_rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            # storage is now our SimpleStorage object
            data = storage._data
            # Calculate the offset and size
            offset = int(storage_offset)
            total_elements = int(np.prod(size)) if len(size) > 0 else len(data)
            tensor_data = data[offset:offset + total_elements]
            return torch.tensor(tensor_data.reshape(size) if len(size) > 0 else tensor_data)
        
        torch._utils._rebuild_tensor_v2 = custom_rebuild
        
        try:
            state_dict = unpickler.load()
        finally:
            torch._utils._rebuild_tensor_v2 = original_rebuild
    
    print(f"Loaded state dict with {len(state_dict)} keys")
    for k, v in state_dict.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
    
    # Create model and load weights
    model = GRUVAE()
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
    
    # Test inference
    dummy_input = torch.randn(1, WINDOW_SIZE, N_FEATURES)
    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)
        recon_error = ((recon - dummy_input) ** 2).mean().item()
        print(f"Inference test: input={dummy_input.shape}, recon={recon.shape}")
        print(f"  Reconstruction error: {recon_error:.4f}")
    
    # Export to ONNX using legacy export
    onnx_path = os.path.join(base_dir, '..', 'gru_vae_best.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")
    
    # Set encoding to handle Unicode
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Use legacy export with opset 17
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=['input'],
        output_names=['reconstruction', 'mu', 'logvar'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'reconstruction': {0: 'batch_size'},
            'mu': {0: 'batch_size'},
            'logvar': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX model exported successfully to {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path)} bytes")


if __name__ == '__main__':
    main()
