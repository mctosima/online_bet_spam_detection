import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import pytz  # Add this import for timezone support
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import wandb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from utils import check_set_gpu

# Import local modules
from ycj_datareader import YouTubeCommentDataset
from model.sentiment_model import BERTSentimentClassifier
from model.lightweight_transformer import LightweightTransformer
from utils import check_set_gpu

import nltk
nltk.download('punkt')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train model for YouTube comment spam detection')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/youtube_chat_jogja_clean.csv', 
                        help='Path to dataset file')
    parser.add_argument('--preprocess', action='store_true', default=True, 
                        help='Apply text preprocessing')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['bert', 'lightweight'], default='lightweight',
                        help='Type of model to use (bert or lightweight)')
    parser.add_argument('--model_name', type=str, default='indobenchmark/indobert-base-p1',
                        help='Pre-trained BERT model name (only for bert model_type)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Lightweight model specific parameters
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for lightweight model')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for lightweight model')
    parser.add_argument('--ff_dim', type=int, default=512,
                        help='Feed-forward dimension for lightweight model')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers for lightweight model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=35,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=6,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='Learning rate scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor by which to reduce learning rate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for loss function')
    
    # Augmentation parameters
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'random', 'synonym', 'delete', 'swap', 'insert', 'backtranslation', 'cascaded'],
                        help='Type of text augmentation to use')
    parser.add_argument('--p_augment', type=float, default=0.5,
                        help='Probability of applying augmentation to a sample')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='ycj_model_outputs',
                        help='Directory to save model outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='youtube-comment-spam',
                       help='Wandb project name')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/username')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Additional name for the wandb run')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_output_dir(base_dir):
    """Create timestamped output directory"""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped directory with Indonesia timezone (UTC+7)
    indonesia_tz = pytz.timezone('Asia/Jakarta')
    timestamp = datetime.now(indonesia_tz).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def plot_training_metrics(train_metrics, val_metrics, metric_name, output_path):
    """Plot training and validation metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Training and Validation {metric_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def get_dataloaders_for_fold(args, fold, current_epoch=0):
    """Create train and validation datasets/dataloaders for the specified fold"""
    # Create datasets
    train_dataset = YouTubeCommentDataset(
        file_path=args.data_path,
        fold=fold,
        n_folds=args.n_folds,
        split="train",
        max_length=args.max_length,
        tokenizer_name=args.model_name if args.model_type == 'bert' else 'bert-base-uncased',
        apply_preprocessing=args.preprocess,
        random_state=args.seed,
        use_local_tokenizer=True,
        augmentation=args.augmentation,
        p_augment=args.p_augment,
        current_epoch=current_epoch
    )
    
    val_dataset = YouTubeCommentDataset(
        file_path=args.data_path,
        fold=fold,
        n_folds=args.n_folds,
        split="val",
        max_length=args.max_length,
        tokenizer_name=args.model_name if args.model_type == 'bert' else 'bert-base-uncased',
        apply_preprocessing=args.preprocess,
        random_state=args.seed,
        use_local_tokenizer=True,
        augmentation='none'  # Never augment validation data
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, train_dataset

def manage_checkpoints(model, epoch, accuracy, fold_output_dir, max_checkpoints=3, best_f1=None):
    """
    Save model checkpoint and maintain only top N checkpoints based on accuracy.
    
    Args:
        model: Model to save
        epoch: Current epoch number
        accuracy: Validation accuracy
        fold_output_dir: Output directory for the current fold
        max_checkpoints: Maximum number of checkpoints to keep
        best_f1: Current best F1 score (to mark best checkpoint)
    
    Returns:
        checkpoint_path: Path to the saved checkpoint
        is_best: Whether this checkpoint is the best so far
    """
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(fold_output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Format checkpoint filename with epoch and accuracy
    checkpoint_name = f"epoch-{epoch+1:02d}_acc-{accuracy:.4f}.pt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    
    # Save current model - use torch.save instead of model.save_model
    try:
        # Save the full model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'model_name': model.model_name if hasattr(model, 'model_name') else None,
        }, checkpoint_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return None, False
        
    # Log success
    print(f"Successfully saved checkpoint to: {os.path.basename(checkpoint_path)}")
    
    # Get all existing checkpoints and their accuracies
    checkpoints = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith(".pt") and file.startswith("epoch-"):
            try:
                # Extract accuracy from filename
                acc = float(file.split("_acc-")[1].split(".pt")[0])
                checkpoints.append((os.path.join(checkpoints_dir, file), acc))
            except (IndexError, ValueError):
                continue
    
    # Sort checkpoints by accuracy (descending)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    # # Remove excess checkpoints (keep only top max_checkpoints)
    # if len(checkpoints) > max_checkpoints:
    #     for cp_path, _ in checkpoints[max_checkpoints:]:
    #         try:
    #             os.remove(cp_path)
    #             print(f"Removed checkpoint: {os.path.basename(cp_path)}")
    #         except OSError as e:
    #             print(f"Error removing checkpoint {cp_path}: {e}")
    
    # Check if this is the best checkpoint (comparing with previous best if provided)
    is_best = best_f1 is None or accuracy > best_f1
    
    # If this is the best checkpoint, create a symlink or copy named "best_model.pt"
    if is_best:
        best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
        try:
            # If a previous best model exists, remove it
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            # Save a copy of the current checkpoint as best_model.pt
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'model_name': model.model_name if hasattr(model, 'model_name') else None,
            }, best_model_path)
            
            print(f"Saved new best model: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"Error creating best model reference: {e}")
    
    # Return path to the current checkpoint and whether it's the best
    return checkpoint_path, is_best

def create_model(args, device):
    """
    Create the appropriate model based on args
    
    Args:
        args: Command line arguments
        device: Device to place model on
        
    Returns:
        Initialized model on the specified device
    """
    if args.model_type == 'bert':
        model = BERTSentimentClassifier(
            model_name=args.model_name,
            num_classes=2,  # Binary classification: 0=normal, 1=spam
            dropout_rate=args.dropout
        )
    else:  # lightweight
        model = LightweightTransformer(
            vocab_size=30522,  # Default vocab size for BERT tokenizer
            max_seq_len=args.max_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            num_classes=2,  # Binary classification
            dropout_rate=args.dropout
        )
    
    # Move model to device
    model.to(device)
    
    return model

def train_fold(fold, args, output_dir, device):
    """Train and evaluate model on a specific fold"""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold+1}/{args.n_folds} with {args.model_type} model")
    print(f"{'='*80}")
    
    # Get data loaders for this fold with epoch 0
    train_loader, val_loader, train_dataset = get_dataloaders_for_fold(args, fold, current_epoch=0)
    
    # Create model
    model = create_model(args, device)
    
    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define scheduler
    total_steps = len(train_loader) * args.epochs
    
    # Different learning rate behavior based on model type
    if args.model_type == 'bert':
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
            num_training_steps=total_steps
        )
    else:
        # For lightweight model, use a simpler scheduler or None
        warmup_scheduler = None
    
    # Add ReduceLROnPlateau scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # since we're tracking f1 score
        factor=args.lr_factor,
        patience=args.lr_patience
    )
    
    # Define loss function
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Reference the appropriate Model class methods for training and evaluation
    if args.model_type == 'bert':
        train_method = BERTSentimentClassifier.train_model
        eval_method = BERTSentimentClassifier.evaluate_model
        compute_metrics = BERTSentimentClassifier.compute_metrics
        model_class = BERTSentimentClassifier
    else:
        train_method = LightweightTransformer.train_model
        eval_method = LightweightTransformer.evaluate_model
        compute_metrics = LightweightTransformer.compute_metrics
        model_class = LightweightTransformer
        
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    learning_rates = []
    best_model_path = None
    
    fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Use trange for epoch progress tracking
    epochs_iter = trange(args.epochs, desc=f"Fold {fold+1}")
    for epoch in epochs_iter:
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===\n")
        
        # Update dataset with current epoch for dynamic augmentation
        train_dataset.set_epoch(epoch)
        
        # Recreate dataloader for train set with updated epoch (optional)
        # If your implementation changes batches due to augmentation, you may want to recreate the dataloader
        # Otherwise, just setting the epoch is sufficient
        if args.augmentation != 'none':
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
        
        # Log info about epoch-aware augmentation
        if args.augmentation != 'none':
            print(f"Using epoch-aware augmentation (epoch {epoch+1})")
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training
        train_start_time = time.time()
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Create progress bar for training steps
        train_iter = tqdm(train_loader, desc=f"Training", leave=False)
        for batch in train_iter:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update scheduler
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            # Update progress bar with current loss
            train_iter.set_postfix({'loss': loss.item()})
        
        # Calculate train metrics
        train_loss = train_loss / len(train_loader)
        train_accuracy, train_f1, train_precision, train_recall, train_macro_f1 = compute_metrics(all_labels, all_predictions, zero_division=1)
        
        train_metrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'weighted_f1': train_f1,
            'macro_f1': train_macro_f1,
            'precision': train_precision,
            'recall': train_recall
        }
        train_time = time.time() - train_start_time
        
        # Validation
        val_start_time = time.time()
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        confusion_matrix = np.zeros((2, 2), dtype=int)  # For binary classification
        
        # Create progress bar for validation steps
        val_iter = tqdm(val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update confusion matrix
                for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    confusion_matrix[true, pred] += 1
                
                # Update progress bar with current loss
                val_iter.set_postfix({'loss': loss.item()})
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy, val_f1, val_precision, val_recall, val_macro_f1 = compute_metrics(all_labels, all_predictions, zero_division=1)
        
        val_metrics = {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'weighted_f1': val_f1,
            'macro_f1': val_macro_f1,
            'precision': val_precision,
            'recall': val_recall,
            'confusion_matrix': confusion_matrix
        }
        val_time = time.time() - val_start_time
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_f1s.append(train_metrics['weighted_f1'])
        val_f1s.append(val_metrics['weighted_f1'])
        
        # Update epoch progress bar with metrics
        epochs_iter.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}", 
            'val_f1': f"{val_metrics['weighted_f1']:.4f}"
        })
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['weighted_f1']:.4f}, "
              f"Time: {train_time:.2f}s")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['weighted_f1']:.4f}, "
              f"Time: {val_time:.2f}s")
        print(f"Current learning rate: {current_lr}")
        
        # Save checkpoint for this epoch, passing current best_val_f1 for comparison
        checkpoint_path, is_best = manage_checkpoints(
            model, epoch, val_metrics['accuracy'], fold_output_dir, 
            max_checkpoints=3, best_f1=best_val_f1
        )
        print(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Step LR scheduler after validation
        prev_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step(val_metrics['weighted_f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log if learning rate changed
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr} to {current_lr}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                f"fold_{fold}/train_loss": train_metrics['loss'],
                f"fold_{fold}/train_accuracy": train_metrics['accuracy'],
                f"fold_{fold}/train_weighted_f1": train_metrics['weighted_f1'],
                f"fold_{fold}/train_macro_f1": train_metrics['macro_f1'],
                f"fold_{fold}/val_loss": val_metrics['loss'],
                f"fold_{fold}/val_accuracy": val_metrics['accuracy'],
                f"fold_{fold}/val_weighted_f1": val_metrics['weighted_f1'],
                f"fold_{fold}/val_macro_f1": val_metrics['macro_f1'],
                f"fold_{fold}/learning_rate": current_lr,
                "epoch": epoch,
                # Add global metrics for easier comparison across folds
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "train/weighted_f1": train_metrics['weighted_f1'],
                "train/macro_f1": train_metrics['macro_f1'],
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/weighted_f1": val_metrics['weighted_f1'],
                "val/macro_f1": val_metrics['macro_f1'],
                "learning_rate": current_lr
            })
        
        # Check for improvement
        if val_metrics['weighted_f1'] > best_val_f1:
            best_val_f1 = val_metrics['weighted_f1']
            best_epoch = epoch
            patience_counter = 0
            best_model_path = os.path.join(fold_output_dir, "checkpoints", "best_model.pt")
            
            # Log improvement but don't save files
            print(f"New best model found (F1: {best_val_f1:.4f})")
            
            # Log best metrics to wandb
            if args.use_wandb:
                wandb.log({
                    f"fold_{fold}/best_val_f1": best_val_f1,
                    f"fold_{fold}/best_epoch": best_epoch
                })
                
                # Log confusion matrix as image
                if args.use_wandb:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(val_metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
                    ax.set(xticks=np.arange(2),
                           yticks=np.arange(2),
                           xticklabels=["Normal", "Spam"],
                           yticklabels=["Normal", "Spam"],
                           title=f'Confusion Matrix (Fold {fold+1})',
                           ylabel='True label',
                           xlabel='Predicted label')
                    
                    # Add text annotations to confusion matrix
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, format(val_metrics['confusion_matrix'][i, j], 'd'),
                                    ha="center", va="center", color="white" if val_metrics['confusion_matrix'][i, j] > val_metrics['confusion_matrix'].max() / 2 else "black")
                    
                    plt.colorbar(im)
                    plt.tight_layout()
                    wandb.log({f"fold_{fold}/confusion_matrix": wandb.Image(fig)})
                    plt.close(fig)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'learning_rates': learning_rates,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1
    }
    
    with open(os.path.join(fold_output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
    
    # Create plots
    plot_training_metrics(train_losses, val_losses, 'loss', os.path.join(fold_output_dir, 'loss_curve.png'))
    plot_training_metrics(train_f1s, val_f1s, 'f1 score', os.path.join(fold_output_dir, 'f1_curve.png'))
    
    # Plot learning rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(fold_output_dir, 'lr_curve.png'))
    plt.close()
    
    # Log plots to wandb
    if args.use_wandb:
        wandb.log({
            f"fold_{fold}/loss_curve": wandb.Image(os.path.join(fold_output_dir, 'loss_curve.png')),
            f"fold_{fold}/f1_curve": wandb.Image(os.path.join(fold_output_dir, 'f1_curve.png')),
            f"fold_{fold}/lr_curve": wandb.Image(os.path.join(fold_output_dir, 'lr_curve.png'))
        })
    
    # Load the best model for final evaluation
    print(f"Loading best model from {best_model_path} for final evaluation")
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Create a new model instance of the same type
        best_model = create_model(args, device)
        
        # Load the saved weights
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use the best model for final evaluation
        final_metrics = eval_method(
            model=best_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        print(f"Final evaluation with best model (epoch {checkpoint['epoch']})")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}, F1: {final_metrics['weighted_f1']:.4f}")
        
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using current model for final evaluation instead")
        final_metrics = eval_method(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
    
    return final_metrics, best_val_f1

def log_run_summary(output_dir, run_name, fold_results, best_f1_scores, best_epochs):
    """
    Create a summary log of the training run with metrics for each fold.
    
    Args:
        output_dir: Base output directory
        run_name: Name of the run
        fold_results: List of metrics for each fold
        best_f1_scores: List of best F1 scores for each fold
        best_epochs: List of best epochs for each fold
    """
    # Create runlog directory if it doesn't exist
    runlog_dir = os.path.join(os.path.dirname(output_dir), "runlog")
    os.makedirs(runlog_dir, exist_ok=True)
    
    # Extract run name from output_dir if not provided
    if run_name is None:
        run_name = os.path.basename(output_dir)
    
    # Prepare the log file path
    log_file = os.path.join(runlog_dir, f"{run_name}.csv")
    
    # Create header and data rows
    header = ["Fold", "Best Epoch", "Accuracy", "Weighted F1", "Macro F1", "Best F1"]
    rows = []
    
    for i, (metrics, best_f1, best_epoch) in enumerate(zip(fold_results, best_f1_scores, best_epochs)):
        rows.append([
            i,
            best_epoch + 1,  # +1 because epochs are 0-indexed
            metrics["accuracy"],
            metrics["weighted_f1"],
            metrics["macro_f1"],
            best_f1
        ])
    
    # Add average row
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in fold_results])
    avg_macro_f1 = np.mean([metrics['macro_f1'] for metrics in fold_results])
    avg_weighted_f1 = np.mean([metrics['weighted_f1'] for metrics in fold_results])
    avg_best_f1 = np.mean(best_f1_scores)
    
    rows.append([
        "Average",
        "-",
        avg_accuracy,
        avg_weighted_f1,
        avg_macro_f1,
        avg_best_f1
    ])
    
    # Write to CSV
    with open(log_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Run summary saved to {log_file}")
    
    # Return the path to the log file
    return log_file

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory - include model type and augmentation in the directory name
    model_dir_name = f"{args.output_dir}_{args.model_type}"
    if args.augmentation != 'none':
        model_dir_name += f"_aug_{args.augmentation}"
    output_dir = create_output_dir(model_dir_name)
    print(f"Output directory: {output_dir}")
    
    # Extract run_name from output_dir for logging
    run_name = os.path.basename(output_dir)
    if args.run_name:
        run_name = f"{run_name}_{args.run_name}"
    
    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb with model type and augmentation info
    if args.use_wandb:
        # Use Indonesia timezone (UTC+7) for the timestamp
        indonesia_tz = pytz.timezone('Asia/Jakarta')
        timestamp = datetime.now(indonesia_tz).strftime('%Y%m%d-%H%M')
        wandb_run_name = f"{timestamp}_{args.model_type}"
        if args.augmentation != 'none':
            wandb_run_name += f"_aug_{args.augmentation}"
        if args.run_name:
            wandb_run_name += f"_{args.run_name}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=output_dir
        )
        # Log code file
        wandb.save(os.path.abspath(__file__))
        print(f"Logging to Weights & Biases project: {args.wandb_project}")
    
    # Determine device
    device = check_set_gpu()
    print(f"Using device: {device}")
    
    # Cross-validation
    fold_results = []
    best_f1_scores = []
    best_epochs = []
    
    for fold in range(args.n_folds):
        fold_metrics, best_val_f1 = train_fold(fold, args, output_dir, device)
        fold_results.append(fold_metrics)
        best_f1_scores.append(best_val_f1)
        
        # Find the best epoch from the fold's training history
        with open(os.path.join(output_dir, f"fold_{fold}", "training_history.json"), "r") as f:
            history = json.load(f)
            best_epochs.append(history["best_epoch"])
    
    # Compute average metrics across folds
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in fold_results])
    avg_macro_f1 = np.mean([metrics['macro_f1'] for metrics in fold_results])
    avg_weighted_f1 = np.mean([metrics['weighted_f1'] for metrics in fold_results])
    avg_best_f1 = np.mean(best_f1_scores)
    
    # Print final results
    print("\n" + "="*80)
    print("Cross-Validation Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Macro F1: {avg_macro_f1:.4f}")
    print(f"Average Weighted F1: {avg_weighted_f1:.4f}")
    print(f"Average Best Validation F1: {avg_best_f1:.4f}")
    print("="*80)
    
    # Log final cross-validation results to wandb
    if args.use_wandb:
        wandb.log({
            "final/avg_accuracy": avg_accuracy,
            "final/avg_macro_f1": avg_macro_f1,
            "final/avg_weighted_f1": avg_weighted_f1,
            "final/avg_best_f1": avg_best_f1
        })
    
    # Save cross-validation results
    cv_results = {
        'fold_metrics': [
            {k: v.tolist() if isinstance(v, np.ndarray) else v 
             for k, v in metrics.items()}
            for metrics in fold_results
        ],
        'best_f1_scores': best_f1_scores,
        'best_epochs': best_epochs,
        'avg_accuracy': avg_accuracy,
        'avg_macro_f1': avg_macro_f1,
        'avg_weighted_f1': avg_weighted_f1,
        'avg_best_f1': avg_best_f1
    }
    
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Create and save run summary
    log_file = log_run_summary(output_dir, run_name, fold_results, best_f1_scores, best_epochs)
    
    # Log summary file to wandb if enabled
    if args.use_wandb:
        wandb.save(log_file)
    
    print(f"\nTraining completed for {args.model_type} model. Results saved to {output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
