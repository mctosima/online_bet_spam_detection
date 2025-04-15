import pandas as pd
import numpy as np
import json
import os
import re
import string
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define Indonesian stopwords + common chat words that don't help in classification
stopwords_id = set(stopwords.words('indonesian'))
additional_stopwords = {'yg', 'dgn', 'nya', 'dan', 'di', 'ke', 'ya', 'atau', 'ini', 'itu',
                      'jg', 'dr', 'krn', 'karna', 'untuk', 'utk', 'dlm', 'dalam', 'aja', 'si', 'ya'}
stopwords_id.update(additional_stopwords)

def preprocess_text(text):
    """Preprocess text for Bahasa Indonesia comments"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords_id]
    
    # Join tokens back
    text = ' '.join(tokens)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_output_folder():
    """Create output folder if it doesn't exist"""
    output_dir = "ycj_model_outputs_svm"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_confusion_matrix(cm, fold, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_fold_{fold}.png")
    plt.close()

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/youtube_chat_jogja_clean.csv')
    
    # Load splits
    with open('./ycj_split.json', 'r') as f:
        splits_data = json.load(f)
    
    # Print structure of splits data for debugging
    print("Structure of splits data:", type(splits_data))
    print("Keys in splits data:", splits_data.keys())
    
    # Create output directory
    output_dir = create_output_folder()
    
    # Initialize results storage
    results = []
    all_predictions = []
    all_true_labels = []
    
    start_time = time.time()
    
    # Check if fold_indices exists
    if 'fold_indices' in splits_data:
        fold_indices = splits_data['fold_indices']
        print(f"Fold indices type: {type(fold_indices)}")
        
        # Handle fold_indices as a dictionary
        if isinstance(fold_indices, dict):
            print(f"Fold indices keys: {fold_indices.keys()}")
            
            # Fix: Include keys that start with 'fold_' 
            fold_keys = sorted([k for k in fold_indices.keys()])
            print(f"Processing {len(fold_keys)} folds")
            
            for fold_idx, fold_key in enumerate(tqdm(fold_keys), 1):
                print(f"\nProcessing Fold {fold_idx} (key: {fold_key})...")
                fold_data = fold_indices[fold_key]
                
                # Extract train and test indices - use train_indices and val_indices keys
                if isinstance(fold_data, dict):
                    # Check for different possible key names
                    if 'train_indices' in fold_data and 'val_indices' in fold_data:
                        train_idx = fold_data['train_indices']
                        test_idx = fold_data['val_indices']
                    elif 'train' in fold_data and 'test' in fold_data:
                        train_idx = fold_data['train']
                        test_idx = fold_data['test']
                    else:
                        print(f"Warning: Fold data doesn't contain recognized train/test keys. Keys found: {fold_data.keys()}")
                        continue
                else:
                    print(f"Warning: Fold data is not a dictionary. Type: {type(fold_data)}")
                    continue
                
                # Validate indices
                if not train_idx or not test_idx:
                    print(f"Warning: Empty train or test indices for fold {fold_idx}. Skipping...")
                    continue
                
                # Split data
                try:
                    train_data = df.iloc[train_idx]
                    test_data = df.iloc[test_idx]
                except Exception as e:
                    print(f"Error splitting data: {e}")
                    print(f"Train indices type: {type(train_idx)}, Test indices type: {type(test_idx)}")
                    print(f"Sample train indices: {train_idx[:5] if hasattr(train_idx, '__getitem__') else train_idx}")
                    continue
                
                # Preprocess text
                print("Preprocessing text...")
                train_data['processed_message'] = train_data['message'].apply(preprocess_text)
                test_data['processed_message'] = test_data['message'].apply(preprocess_text)
                
                X_train = train_data['processed_message']
                y_train = train_data['label']
                X_test = test_data['processed_message']
                y_test = test_data['label']
                
                # Create pipeline with TF-IDF and SVM
                print("Training SVM model...")
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                    ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True))
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Store for final metrics
                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Get per-class F1 and find best F1
                report = classification_report(y_test, y_pred, output_dict=True)
                class_f1s = {k: v['f1-score'] for k, v in report.items() if k.isdigit()}
                best_f1 = max(class_f1s.values()) if class_f1s else 0
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cm, fold_idx, output_dir)
                
                # Save detailed fold results
                fold_report = pd.DataFrame(report).transpose()
                fold_report.to_csv(f"{output_dir}/fold_{fold_idx}_report.csv")
                
                # Store results
                results.append({
                    'fold': fold_idx,
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'best_f1': best_f1
                })
                
                print(f"Fold {fold_idx} results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Macro F1: {macro_f1:.4f}")
                print(f"  Weighted F1: {weighted_f1:.4f}")
                print(f"  Best F1: {best_f1:.4f}")
        # Handle fold_indices as a list
        elif isinstance(fold_indices, list):
            print(f"Processing {len(fold_indices)} folds")
            
            for fold_num, fold_data in enumerate(tqdm(fold_indices), 1):
                print(f"\nProcessing Fold {fold_num}...")
                
                # Extract train and test indices based on structure
                if isinstance(fold_data, dict) and 'train' in fold_data and 'test' in fold_data:
                    train_idx = fold_data['train']
                    test_idx = fold_data['test']
                else:
                    print(f"Warning: Fold data doesn't contain train/test keys: {fold_data}")
                    continue
                
                # Rest of processing is the same
                # Split data
                try:
                    train_data = df.iloc[train_idx]
                    test_data = df.iloc[test_idx]
                except Exception as e:
                    print(f"Error splitting data: {e}")
                    print(f"Train indices type: {type(train_idx)}, Test indices type: {type(test_idx)}")
                    print(f"Sample train indices: {train_idx[:5] if hasattr(train_idx, '__getitem__') else train_idx}")
                    continue
                
                # Preprocess text
                print("Preprocessing text...")
                train_data['processed_message'] = train_data['message'].apply(preprocess_text)
                test_data['processed_message'] = test_data['message'].apply(preprocess_text)
                
                X_train = train_data['processed_message']
                y_train = train_data['label']
                X_test = test_data['processed_message']
                y_test = test_data['label']
                
                # Create pipeline with TF-IDF and SVM
                print("Training SVM model...")
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                    ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True))
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Store for final metrics
                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Get per-class F1 and find best F1
                report = classification_report(y_test, y_pred, output_dict=True)
                class_f1s = {k: v['f1-score'] for k, v in report.items() if k.isdigit()}
                best_f1 = max(class_f1s.values()) if class_f1s else 0
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cm, fold_num, output_dir)
                
                # Save detailed fold results
                fold_report = pd.DataFrame(report).transpose()
                fold_report.to_csv(f"{output_dir}/fold_{fold_num}_report.csv")
                
                # Store results
                results.append({
                    'fold': fold_num,
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'best_f1': best_f1
                })
                
                print(f"Fold {fold_num} results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Macro F1: {macro_f1:.4f}")
                print(f"  Weighted F1: {weighted_f1:.4f}")
                print(f"  Best F1: {best_f1:.4f}")
        else:
            print(f"Error: Unrecognized structure for fold_indices: {type(fold_indices)}")
            return
    else:
        print("Error: Could not find 'fold_indices' in the split data")
        return
    
    # Check if we have any results
    if not all_predictions:
        print("Error: No predictions were made. Check the fold indices and dataset.")
        return
    
    # Calculate final metrics on all predictions
    final_accuracy = accuracy_score(all_true_labels, all_predictions)
    final_macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    final_weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    # Final report
    final_report = classification_report(all_true_labels, all_predictions, output_dict=True)
    class_f1s = {k: v['f1-score'] for k, v in final_report.items() if k.isdigit()}
    final_best_f1 = max(class_f1s.values()) if class_f1s else 0
    
    # Final confusion matrix
    final_cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Final Confusion Matrix (All Folds)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_confusion_matrix.png")
    plt.close()
    
    # Add final results
    results.append({
        'fold': 'Final',
        'accuracy': final_accuracy,
        'macro_f1': final_macro_f1,
        'weighted_f1': final_weighted_f1,
        'best_f1': final_best_f1
    })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/svm_performance_results.csv", index=False)
    
    # Save final detailed report
    final_report_df = pd.DataFrame(final_report).transpose()
    final_report_df.to_csv(f"{output_dir}/final_classification_report.csv")
    
    elapsed_time = time.time() - start_time
    print(f"\nModel training and evaluation completed in {elapsed_time:.2f} seconds")
    print(f"Final results:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  Macro F1: {final_macro_f1:.4f}")
    print(f"  Weighted F1: {final_weighted_f1:.4f}")
    print(f"  Best F1: {final_best_f1:.4f}")
    print(f"All results and reports saved to {output_dir}/")

if __name__ == "__main__":
    main()
