import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

class BERTSentimentClassifier(nn.Module):
    """
    BERT-based model for sentiment classification (rating prediction)
    """
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased", 
                 num_classes: int = 5,
                 dropout_rate: float = 0.3):
        """
        Initialize BERT sentiment classifier
        
        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of output classes (rating 1-5)
            dropout_rate: Dropout probability for classification head
        """
        super(BERTSentimentClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size from the BERT configuration
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token ids from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Logits for each class
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    @staticmethod
    def train_model(model: nn.Module, 
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict:
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
    def evaluate_model(model: nn.Module, 
                      dataloader: torch.utils.data.DataLoader,
                      criterion: nn.Module,
                      device: torch.device) -> Dict:
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
    
    def save_model(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'model_name': self.bert.config._name_or_path,
                'num_classes': self.classifier[-1].out_features,
                'dropout_rate': self.classifier[0].p
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> nn.Module:
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
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
