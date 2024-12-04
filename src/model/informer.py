"""
Informer model implementation for time series forecasting.

Dependencies:
- torch>=2.0.1
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .layers import ProbAttention, AttentionLayer

class ConvLayer(nn.Module):
    """Convolutional Layer for downsampling"""

    def __init__(self, c_in: int):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    """Encoder layer with self-attention mechanism"""

    def __init__(
            self,
            attention: nn.Module,
            d_model: int,
            d_ff: Optional[int] = None,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x+y), attn

class Encoder(nn.Module):
    """Informer encoder"""

    def __init__(
            self,
            attn_layers: nn.ModuleList,
            conv_layers: Optional[nn.ModuleList] = None,
            norm_layer: Optional[nn.Module] = None
    ):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm = norm_layer

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Informer(nn.Module):
    """
    Informer model for time series forecasting
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int = 512,
            n_heads: int = 8,
            e_layers: int = 3,
            d_ff: int = 512,
            dropout: float = 0.0,
            activation: str = 'gelu',
            distil: bool = True,
            prediction_window: int = 48,  
    ):
        """
        Initialize Informer model

        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function type
            distil: Whether to use distilling in encoder
        """
        super(Informer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_window = prediction_window

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)

        # Encoder
        encoder_layers = []
        conv_layers = []

        for i in range(e_layers):
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            )
            if i < e_layers-1 and distil:
                conv_layers.append(ConvLayer(d_model))

        self.encoder = Encoder(
            nn.ModuleList(encoder_layers),
            nn.ModuleList(conv_layers) if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Output projection
        self.projection = nn.Linear(d_model, output_dim, bias=True)

    def forward(self, x: torch.Tensor, enc_self_mask: Optional[torch.Tensor] = None):
        print(f"\nModel Forward Debug:")
        print(f"Input shape: {x.shape}")

        # Process in smaller chunks if input is too large
        batch_size = x.size(0)
        chunk_size = 16  # Process 16 samples at a time
        outputs = []
        
        for i in range(0, batch_size, chunk_size):
            chunk = x[i:i+chunk_size]
            
            # Encoding
            enc_out = self.enc_embedding(chunk)
            
            # Encoder
            enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
            
            # Output projection
            chunk_output = self.projection(enc_out)
            
            # Store chunk output
            outputs.append(chunk_output)

            # Clean up memory
            del enc_out
            torch.cuda.empty_cache()

        # Combine chunks
        output = torch.cat(outputs, dim=0)
        
        # Debug prints
        print(f"After projection shape: {output.shape}")
        
        # Ensure output matches target length
        output = output[:, :self.prediction_window-1, :]
        print(f"Final output shape: {output.shape}")
        print(f"Prediction window size: {self.prediction_window}")
        print("=" * 50)
        
        # Return empty list for attns to save memory
        return output, []

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)