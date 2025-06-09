import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Self‐attention for sequence outputs of an LSTM."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor):
        """
        lstm_output: (batch_size, seq_len, hidden_dim)
        returns:
          - context_vector: (batch_size, hidden_dim)
          - attention_weights: (batch_size, seq_len)
        """
        scores = self.attention(lstm_output).squeeze(-1)  
        weights = F.softmax(scores, dim=1)                
        context = torch.bmm(
            weights.unsqueeze(1),    
            lstm_output               
        ).squeeze(1)                  
        return context, weights


class EnhancedBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        emb_matrix: torch.Tensor,
        feature_dim: int      = 16,
        hidden_dim: int       = 128,
        num_layers: int       = 2,
        num_classes: int      = 2,
        dropout: float        = 0.5,
        use_attention: bool   = True
    ):
        super().__init__()
        num_embeddings, emb_dim = emb_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(emb_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0
        )
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        final_in = hidden_dim * 2 + 32
        self.fc = nn.Sequential(
            nn.Linear(final_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor):
        """
        x: (batch_size, seq_len) token indices
        features: (batch_size, feature_dim) numeric features
        returns: (logits, attention_weights)
        """
        emb = self.embedding(x)  
        lstm_out, _ = self.bilstm(emb) 
        if self.use_attention:
            context, attn_w = self.attention(lstm_out) 
            text_feat = self.dropout(context)
        else:
            text_feat = self.dropout(lstm_out[:, -1, :])
            attn_w   = None
        feat_out = self.feature_layer(features)  
        combined = torch.cat([text_feat, feat_out], dim=1)
        logits   = self.fc(combined)           
        return logits, attn_w