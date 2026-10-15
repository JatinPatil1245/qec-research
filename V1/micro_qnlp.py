# micro_qnlp.py
import torch
import torch.nn as nn

from quantum_embedding import HybridQuantumEmbedding 

class MicroQuantumTransformer(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=8, num_qubits=4, num_classes=2):
        super().__init__()
        
        # Classical Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Classical Self-Attention Head
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        
        # Quantum Feed-Forward Block (Imported from your previous work)
        # We pass embed_dim (8) as the classical_dim.
        self.quantum_ffn = HybridQuantumEmbedding(classical_dim=embed_dim, num_qubits=num_qubits)
        
        # 4. Final Classification Head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)
        
        # Classical Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # Global Average Pooling (collapse sequence into a single vector)
        x_pooled = x.mean(dim=1)
        
        # Quantum Activation (Routes through your zz_feature_map and real_amplitudes)
        q_features = self.quantum_ffn(x_pooled)
        
        # Final logits
        return self.classifier(q_features)