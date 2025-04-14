import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import wandb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

# Import local modules
from shopee_datareader_simple import ShopeeCommentDataset
from sentiment_model import BERTSentimentClassifier
from utils import check_set_gpu

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train BERT for Shopee sentiment analysis')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='dataset_shopee.xlsx', 
                        help='Path to dataset file')
    parser.add_argument('--language', type=str, default='indonesian', 
                        choices=['indonesian', 'english'],
                        help='Dataset language')
    parser.add_argument('--preprocess', action='store_true', default=True, 
                        help='Apply text preprocessing')
    parser.add_argument('--augment', action='store_true', 
                        help='Apply text augmentation')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='indolem/indobert-base-uncased',
                        help='Pre-trained BERT model name')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Learning rate scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor by which to reduce learning rate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for loss function')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                        help='Directory to save model outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='shopee-sentiment',
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
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

def get_dataloaders_for_fold(args, fold):
    """Create train and validation datasets/dataloaders for the specified fold"""
    # Create datasets
    train_dataset = ShopeeCommentDataset(
        file_path=args.data_path,
        fold=fold,
        n_folds=args.n_folds,
        split="train",
        max_length=args.max_length,
        tokenizer_name=args.model_name,
        apply_preprocessing=args.preprocess,
        random_state=args.seed
    )
    
    val_dataset = ShopeeCommentDataset(
        file_path=args.data_path,
        fold=fold,
        n_folds=args.n_folds,
        split="val",
        max_length=args.max_length,
        tokenizer_name=args.model_name,
        apply_preprocessing=args.preprocess,
        random_state=args.seed
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

def train_fold(fold, args, output_dir, device):
    """Train and evaluate model on a specific fold"""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold+1}/{args.n_folds}")
    print(f"{'='*80}")
    
    # Get data loaders for this fold
    train_loader, val_loader, train_dataset = get_dataloaders_for_fold(args, fold)
    
    # Create model
    model = BERTSentimentClassifier(
        model_name=args.model_name,
        num_classes=5,  # Ratings 1-5
        dropout_rate=args.dropout
    )
    model.to(device)
    
    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
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
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    learning_rates = []
    
    fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Use trange for epoch progress tracking
    epochs_iter = trange(args.epochs, desc=f"Fold {fold+1}")
    for epoch in epochs_iter:
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
        train_accuracy, train_f1, train_precision, train_recall, train_macro_f1 = BERTSentimentClassifier.compute_metrics(all_labels, all_predictions, zero_division=1)
        
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
        confusion_matrix = np.zeros((5, 5), dtype=int)  # For 5 classes (ratings 1-5)
        
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
        val_accuracy, val_f1, val_precision, val_recall, val_macro_f1 = BERTSentimentClassifier.compute_metrics(all_labels, all_predictions, zero_division=1)
        
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
            
            # Save best model
            model.save_model(os.path.join(fold_output_dir, "best_model.pt"))
            print(f"New best model saved (F1: {best_val_f1:.4f})")
            
            # Save confusion matrix
            np.savetxt(
                os.path.join(fold_output_dir, "confusion_matrix.csv"),
                val_metrics['confusion_matrix'],
                delimiter=","
            )
            
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
                    ax.set(xticks=np.arange(5),
                           yticks=np.arange(5),
                           xticklabels=[1, 2, 3, 4, 5],
                           yticklabels=[1, 2, 3, 4, 5],
                           title=f'Confusion Matrix (Fold {fold+1})',
                           ylabel='True label',
                           xlabel='Predicted label')
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
    
    # Load best model for final evaluation
    best_model = BERTSentimentClassifier.load_model(
        os.path.join(fold_output_dir, "best_model.pt"),
        device
    )
    
    # Final evaluation
    final_metrics = BERTSentimentClassifier.evaluate_model(
        model=best_model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )
    
    return final_metrics, best_val_f1

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb
    if args.use_wandb:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        run_name = f"{timestamp}_"
        if args.run_name:
            run_name += args.run_name
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            dir=output_dir
        )
        # Log code file
        wandb.save(os.path.abspath(__file__))
        print(f"Logging to Weights & Biases project: {args.wandb_project}")
    
    # Determine device
    device = check_set_gpu()
    
    # Cross-validation
    fold_results = []
    best_f1_scores = []
    
    for fold in range(args.n_folds):
        fold_metrics, best_val_f1 = train_fold(fold, args, output_dir, device)
        fold_results.append(fold_metrics)
        best_f1_scores.append(best_val_f1)
    
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
        'avg_accuracy': avg_accuracy,
        'avg_macro_f1': avg_macro_f1,
        'avg_weighted_f1': avg_weighted_f1,
        'avg_best_f1': avg_best_f1
    }
    
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\nTraining completed. Results saved to {output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
