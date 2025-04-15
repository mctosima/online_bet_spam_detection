import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torchinfo import summary

class SimpleCNNClassifier(nn.Module):
    """
    A simple CNN-based text classifier for benchmark comparison
    """
    def __init__(self, 
                 vocab_size=30522,  # Default vocab size for BERT tokenizer
                 embed_dim=100,
                 num_filters=100,
                 filter_sizes=[3, 4, 5],
                 max_seq_len=128,
                 num_classes=2,
                 dropout_rate=0.5):
        """
        Initialize simple CNN text classifier
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_filters: Number of convolutional filters for each filter size
            filter_sizes: List of filter sizes for the convolutions
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Store model name for compatibility with other models
        self.model_name = "simple_cnn"
        
        # Embedding layer - converts token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(len(filter_sizes) * num_filters, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask (not used in this model but kept for API compatibility)
        
        Returns:
            Logits for each class
        """
        # Get embeddings - shape: (batch_size, seq_len, embed_dim)
        x = self.embedding(input_ids)
        
        # Transpose for convolution - shape: (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU
            conv_output = F.relu(conv(x))
            # Apply max pooling over time (sequence length)
            pooled = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate pooled outputs from each filter size
        x = torch.cat(conv_outputs, dim=1)
        
        # Pass through the classifier
        logits = self.classifier(x)
        
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
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
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
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids)
                
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
            'num_filters': self.convs[0].out_channels,
            'filter_sizes': [conv.kernel_size[0] for conv in self.convs],
            'num_classes': self.classifier[-1].out_features,
            'dropout_rate': self.classifier[0].p,
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
            num_filters=config['num_filters'],
            filter_sizes=config['filter_sizes'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    # Create a sample model instance
    model = SimpleCNNClassifier()
    
    # Define sample input dimensions
    batch_size = 8
    seq_length = 128
    
    # Print model summary using torchinfo
    summary_result = summary(
        model,
        input_data={
            "input_ids": torch.randint(0, 30522, (batch_size, seq_length))
        },
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    
    # Calculate total parameters and print
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel summary printed above. Sample forward pass with batch size:", batch_size)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nThis is a simple CNN model for text classification that can be used as a baseline benchmark.")
    print("It's much smaller than transformer-based models like BERT or the lightweight transformer.")
