"""
EEG-Conformer: State-of-the-Art Model for EEG Seizure Detection

Architecture combines:
1. Temporal Convolutional Block - Captures local temporal patterns
2. Spatial Convolutional Block - Models cross-channel relationships  
3. Conformer Block (Transformer + Convolution) - Long-range dependencies
4. Classification Head - Final prediction

Reference: EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Temporal Convolutional Block for initial feature extraction
    Converts raw EEG into patch embeddings
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # First temporal convolution - large kernel for low-frequency patterns
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(1, config.temp_conv_filters[0], 
                     kernel_size=(1, config.temp_conv_kernels[0]),
                     padding=(0, config.temp_conv_kernels[0] // 2)),
            nn.BatchNorm2d(config.temp_conv_filters[0]),
            nn.GELU(),
        )
        
        # Second temporal convolution - smaller kernel for high-frequency patterns
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(config.temp_conv_filters[0], config.temp_conv_filters[1],
                     kernel_size=(1, config.temp_conv_kernels[1]),
                     padding=(0, config.temp_conv_kernels[1] // 2)),
            nn.BatchNorm2d(config.temp_conv_filters[1]),
            nn.GELU(),
        )
        
        # Spatial convolution - learn cross-channel relationships
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(config.temp_conv_filters[1], config.spatial_conv_filters,
                     kernel_size=(config.num_channels, 1)),
            nn.BatchNorm2d(config.spatial_conv_filters),
            nn.GELU(),
            nn.Dropout(config.spatial_dropout),
        )
        
        # Pooling to reduce temporal dimension
        self.pool = nn.AvgPool2d(kernel_size=(1, config.temp_pool_size))
        
        # Calculate output sequence length
        self.seq_len = config.sequence_length // config.temp_pool_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, time) -> (batch, seq_len, embed_dim)
        """
        # Add channel dimension for 2D conv: (B, 1, C, T)
        x = x.unsqueeze(1)
        
        # Temporal convolutions
        x = self.temp_conv1(x)
        x = self.temp_conv2(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, filters, 1, T)
        
        # Pooling
        x = self.pool(x)  # (B, filters, 1, T//pool_size)
        
        # Reshape to sequence: (B, T//pool_size, filters)
        x = x.squeeze(2).transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding"""
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, dim)"""
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class ConvolutionModule(nn.Module):
    """
    Convolution module in Conformer block
    Captures local context with depthwise separable convolutions
    """
    
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(dim)
        
        self.conv = nn.Sequential(
            # Pointwise conv
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            
            # Depthwise conv
            nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                     padding=kernel_size // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            
            # Pointwise conv
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim)"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, seq_len, dim)
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = int(dim * mlp_ratio)
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    """
    Conformer Block: Combines self-attention with convolutions
    Structure: FFN -> MHSA -> Conv -> FFN (with residual connections)
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-norm for attention
        self.attn_norm = nn.LayerNorm(dim)
        
        # First feed-forward (half residual)
        self.ff1 = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            dim, heads=heads, dropout=dropout, attention_dropout=attention_dropout
        )
        
        # Convolution module
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        
        # Second feed-forward (half residual)
        self.ff2 = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First FFN (half residual)
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        x = x + self.attn(self.attn_norm(x))
        
        # Convolution
        x = x + self.conv(x)
        
        # Second FFN (half residual)
        x = x + 0.5 * self.ff2(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class EEGConformer(nn.Module):
    """
    EEG-Conformer: State-of-the-Art Model for EEG Analysis
    
    Combines convolutional feature extraction with transformer-based
    sequence modeling for optimal seizure detection performance.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Patch embedding (temporal + spatial convolution)
        self.patch_embed = PatchEmbedding(config)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.transformer_dim,
            max_len=self.patch_embed.seq_len,
            dropout=config.transformer_dropout
        )
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=config.transformer_dim,
                heads=config.transformer_heads,
                mlp_ratio=config.transformer_mlp_ratio,
                conv_kernel_size=31,
                dropout=config.transformer_dropout,
                attention_dropout=config.attention_dropout
            )
            for _ in range(config.transformer_depth)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, channels, time) - Raw EEG input
            
        Returns:
            logits: (batch, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, seq_len, dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Global pooling: (B, seq_len, dim) -> (B, dim)
        x = x.transpose(1, 2)  # (B, dim, seq_len)
        x = self.global_pool(x)  # (B, dim)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Extract attention maps for visualization"""
        attention_maps = []
        
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        
        for block in self.conformer_blocks:
            # Get attention weights
            B, N, C = x.shape
            qkv = block.attn.qkv(block.attn_norm(x))
            qkv = qkv.reshape(B, N, 3, block.attn.heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = F.softmax(attn, dim=-1)
            attention_maps.append(attn.detach())
            
            # Continue forward pass
            x = block(x)
        
        return attention_maps
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with probabilities
        
        Returns:
            predictions: (batch,) - Predicted class
            probabilities: (batch, num_classes) - Class probabilities
        """
        self.eval()
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config: ModelConfig) -> EEGConformer:
    """Create EEG-Conformer model"""
    model = EEGConformer(config)
    print(f"\nModel created: EEG-Conformer")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Input: ({config.num_channels}, {config.sequence_length})")
    print(f"  Output: {config.num_classes} classes")
    return model


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    model = create_model(config.model)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.model.num_channels, config.model.sequence_length)
    
    print(f"\nInput shape: {x.shape}")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")
