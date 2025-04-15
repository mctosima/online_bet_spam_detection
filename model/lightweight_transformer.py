import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torchinfo import summary

class MultiHeadSelfAttention(nn.Module):
    """
    Simple multi-head self-attention implementation
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if mask is not None:
            # Convert 2D mask (batch_size, seq_len) to 4D mask (batch_size, 1, 1, seq_len)
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # Expand mask to match scores dimensions (batch_size, num_heads, seq_len, seq_len)
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            
            # Apply mask
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(context)
        return output

class TransformerEncoderLayer(nn.Module):
    """
    Lightweight transformer encoder layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class LightweightTransformer(nn.Module):
    """
    Lightweight transformer model for text classification
    """
    def __init__(self, 
                 vocab_size=30522,  # Default vocab size for BERT tokenizer
                 max_seq_len=128,
                 embed_dim=256,
                 num_heads=4,
                 ff_dim=512,
                 num_layers=3,
                 num_classes=5,
                 dropout_rate=0.1):
        """
        Initialize lightweight transformer for text classification
        
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create transformer encoder layers
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate) 
             for _ in range(num_layers)]
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask
        
        Returns:
            Logits for each class
        """
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding[:, :input_ids.size(1), :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Use the [CLS] token representation (first token) for classification
        cls_token = x[:, 0, :]
        
        # Pass through the classification head
        logits = self.classifier(cls_token)
        
        return logits
    
    @staticmethod
    def train_model(model, dataloader, optimizer, criterion, device, scheduler=None):
        """
        Train the model for one epoch
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on (CPU or GPU)
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            # Store predictions and true labels for metrics calculation
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Return metrics
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
    
    @staticmethod
    def evaluate_model(model, dataloader, criterion, device):
        """
        Evaluate the model
        
        Args:
            model: Model to evaluate
            dataloader: Evaluation data loader
            criterion: Loss function
            device: Device to evaluate on (CPU or GPU)
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Store predictions and true labels
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Return metrics
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': conf_matrix
        }
    
    @staticmethod
    def compute_metrics(true_labels, predictions, zero_division=1):
        """
        Compute evaluation metrics from true labels and predictions
        
        Args:
            true_labels: List of true labels
            predictions: List of predicted labels
            zero_division: Value to return when there is a zero division (0 or 1)
            
        Returns:
            Tuple of (accuracy, weighted_f1, precision, recall, macro_f1)
        """
        accuracy = accuracy_score(true_labels, predictions)
        weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=zero_division)
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=zero_division)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=zero_division)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=zero_division)
        
        return accuracy, weighted_f1, precision, recall, macro_f1
    
    def save_model(self, path):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        model_config = {
            'vocab_size': self.embedding.num_embeddings,
            'embed_dim': self.embedding.embedding_dim,
            'num_heads': self.transformer_blocks[0].self_attn.num_heads,
            'ff_dim': self.transformer_blocks[0].ff[0].out_features,
            'num_layers': len(self.transformer_blocks),
            'num_classes': self.classifier[-1].out_features,
            'dropout_rate': self.dropout.p,
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': model_config
        }, path)
    
    @classmethod
    def load_model(cls, path, device):
        """
        Load a saved model
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    # Create a sample model instance
    model = LightweightTransformer()
    
    # Define sample input dimensions
    batch_size = 8
    seq_length = 128
    
    # Print model summary using torchinfo
    summary_result = summary(
        model,
        input_data={
            "input_ids": torch.randint(0, 30522, (batch_size, seq_length)),
            "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long)
        },
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    
    print("\nModel summary printed above. Sample forward pass with batch size:", batch_size)
    print("\nThis model has significantly fewer parameters than DistilBERT (66M)")
