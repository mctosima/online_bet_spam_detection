import pandas as pd
import numpy as np
import json
import os
import re
import string
import random
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
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
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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

class TextAugmenter:
    """Class for text augmentation techniques"""
    
    def __init__(self, p_augment=0.5, language='indonesian'):
        """
        Initialize the text augmenter
        
        Args:
            p_augment: Probability of applying augmentation to a sample
            language: Language of the text (impacts synonym selection)
        """
        self.p_augment = p_augment
        self.language = language
        
        # Load stopwords
        if language == 'indonesian':
            self.stopwords = set(stopwords.words('indonesian'))
        else:
            self.stopwords = set(stopwords.words('english'))
            
    def get_synonyms(self, word):
        """Get synonyms for a word (simplified for Indonesian)"""
        synonyms = []
        
        # For Indonesian, we'll use a simple approach with WordNet
        # This is not optimal for Indonesian but serves as a placeholder
        synsets = wn.synsets(word)
        if synsets:
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in synonyms:
                        synonyms.append(synonym)
        
        return synonyms
    
    def synonym_replacement(self, words, n_words=1):
        """Replace n random words with their synonyms"""
        if not words:
            return words
            
        # Filter out stopwords
        new_words = words.copy()
        random_word_indices = [i for i, word in enumerate(words) if word not in self.stopwords]
        
        # If no valid words found, return original
        if not random_word_indices:
            return words
        
        # Determine number of words to replace
        n_words = min(n_words, len(random_word_indices))
        
        # Replace random words with synonyms
        random.shuffle(random_word_indices)
        for i in range(n_words):
            idx = random_word_indices[i]
            word = words[idx]
            synonyms = self.get_synonyms(word)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return new_words
    
    def random_deletion(self, words, p_delete=0.1):
        """Randomly delete words from the text with probability p_delete"""
        if not words:
            return words
            
        # Keep at least one word
        if len(words) == 1:
            return words
            
        # Randomly delete words
        new_words = []
        for word in words:
            # Random number for this word
            r = random.uniform(0, 1)
            if r > p_delete:
                new_words.append(word)
                
        # If all words were deleted, keep one random word
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return new_words
    
    def random_swap(self, words, n_swaps=1):
        """Randomly swap words n times"""
        if not words or len(words) < 2:
            return words
            
        new_words = words.copy()
        
        # Determine number of swaps
        n_swaps = min(n_swaps, len(words) // 2)  # Cap at half the length
        
        for _ in range(n_swaps):
            # Get random indices to swap
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            # Swap words
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return new_words
    
    def random_insertion(self, words, n_inserts=1):
        """Randomly insert synonyms n times"""
        if not words:
            return words
            
        new_words = words.copy()
        
        # Filter out stopwords for finding random words
        candidates = [word for word in words if word not in self.stopwords]
        
        # If no valid words found, return original
        if not candidates:
            return words
        
        # Determine number of insertions
        n_inserts = min(n_inserts, len(words))
        
        for _ in range(n_inserts):
            # Get random word and its synonyms
            random_word = random.choice(candidates)
            synonyms = self.get_synonyms(random_word)
            
            # If synonyms found, insert one at a random position
            if synonyms:
                synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(new_words))
                new_words.insert(random_idx, synonym)
            
        return new_words
    
    def cascaded_augmentation(self, text):
        """
        Apply multiple augmentation techniques in sequence:
        1. Synonym replacement
        2. Random deletion
        3. Random swap
        4. Random insertion
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text after applying all techniques in sequence
        """
        # Tokenize text
        words = word_tokenize(text)
        
        # If text is too short, don't augment
        if len(words) < 3:
            return text
        
        # 1. Apply synonym replacement (10-15% of words)
        n_words = max(1, int(len(words) * random.uniform(0.1, 0.15)))
        words = self.synonym_replacement(words, n_words)
        
        # 2. Apply random deletion (with 10% probability)
        words = self.random_deletion(words, p_delete=0.1)
        
        # 3. Apply random swap (1 swap)
        words = self.random_swap(words, n_swaps=1)
        
        # 4. Apply random insertion (1 insert)
        words = self.random_insertion(words, n_inserts=1)
        
        # Rebuild text
        return ' '.join(words)
    
    def augment(self, text, method='random'):
        """
        Apply text augmentation
        
        Args:
            text: Text to augment
            method: Augmentation method ('random', 'synonym', 'delete', 'swap', 
                    'insert', 'cascaded', or 'none')
        
        Returns:
            Augmented text
        """
        # Decide whether to augment
        if method == 'none' or random.random() > self.p_augment:
            return text
            
        # Tokenize text
        words = word_tokenize(text)
        
        # If text is too short, don't augment
        if len(words) < 3:
            return text
        
        # Select random method if set to 'random'
        if method == 'random':
            method = random.choice(['synonym', 'delete', 'swap', 'insert', 'cascaded'])
        
        # Apply selected augmentation method
        if method == 'synonym':
            # Replace 10-20% of words with synonyms
            n_words = max(1, int(len(words) * random.uniform(0.1, 0.2)))
            words = self.synonym_replacement(words, n_words)
            return ' '.join(words)
        elif method == 'delete':
            # Delete words with 10-20% probability
            p_delete = random.uniform(0.1, 0.2)
            words = self.random_deletion(words, p_delete)
            return ' '.join(words)
        elif method == 'swap':
            # Swap 1-2 pairs of words
            n_swaps = random.randint(1, 2)
            words = self.random_swap(words, n_swaps)
            return ' '.join(words)
        elif method == 'insert':
            # Insert 1-2 synonyms
            n_inserts = random.randint(1, 2)
            words = self.random_insertion(words, n_inserts)
            return ' '.join(words)
        elif method == 'cascaded':
            # Apply cascaded augmentation (multiple techniques in sequence)
            return self.cascaded_augmentation(text)
        
        # Default return (shouldn't reach here)
        return text
        
def create_augmented_data(data, augmenter, method='random', augment_ratio=1.0, current_epoch=0):
    """
    Create augmented data from original dataset
    
    Args:
        data: DataFrame containing original data
        augmenter: TextAugmenter instance
        method: Augmentation method
        augment_ratio: Ratio of samples to augment (1.0 = all)
        current_epoch: Current epoch number (for epoch-aware augmentation)
    
    Returns:
        DataFrame with original and augmented data
    """
    # Make a copy to avoid modifying the original
    data_copy = data.copy()
    
    # Determine how many samples to augment
    n_augment = int(len(data) * augment_ratio)
    
    # Use epoch as part of the random seed to get different augmentations per epoch
    epoch_seed = 42 + current_epoch
    np.random.seed(epoch_seed)
    
    # Select random samples to augment
    augment_indices = np.random.choice(len(data), size=n_augment, replace=False)
    
    augmented_texts = []
    augmented_labels = []
    
    print(f"Creating {n_augment} augmented samples for epoch {current_epoch}...")
    for idx in tqdm(augment_indices):
        text = data_copy.iloc[idx]['processed_message']
        label = data_copy.iloc[idx]['label']
        
        # Set a deterministic but different seed for each sample and epoch
        sample_epoch_seed = hash(str(idx) + str(current_epoch)) % 10000
        random.seed(sample_epoch_seed)
        
        # Choose augmentation method dynamically if set to 'random'
        if method == 'random':
            # Change available methods based on epoch to create diversity
            epoch_mod = current_epoch % 5
            if epoch_mod == 0:
                aug_method = random.choice(['synonym', 'delete'])
            elif epoch_mod == 1:
                aug_method = random.choice(['swap', 'insert'])
            elif epoch_mod == 2:
                aug_method = 'cascaded'
            elif epoch_mod == 3:
                aug_method = random.choice(['delete', 'swap'])
            else:
                aug_method = random.choice(['synonym', 'delete', 'swap', 'insert', 'cascaded'])
        else:
            aug_method = method
        
        # Apply augmentation
        augmented_text = augmenter.augment(text, method=aug_method)
        
        augmented_texts.append(augmented_text)
        augmented_labels.append(label)
        
        # Reset random seed
        random.seed()
    
    # Reset numpy random seed
    np.random.seed()
    
    # Create DataFrame with augmented data
    augmented_df = pd.DataFrame({
        'message': data_copy.iloc[augment_indices]['message'].values,
        'processed_message': augmented_texts,
        'label': augmented_labels,
        'is_augmented': 1  # Flag to identify augmented samples
    })
    
    # Add is_augmented column to original data
    data_copy['is_augmented'] = 0
    
    # Combine original and augmented data
    combined_df = pd.concat([data_copy, augmented_df], ignore_index=True)
    
    print(f"Original data: {len(data)} samples")
    print(f"Augmented data: {len(augmented_df)} samples")
    print(f"Combined data: {len(combined_df)} samples")
    
    return combined_df

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

def main(augmentation='none', p_augment=0.5, augment_ratio=1.0, num_epochs=5):
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
    
    # Initialize text augmenter if needed
    if augmentation != 'none':
        augmenter = TextAugmenter(p_augment=p_augment)
        print(f"Using text augmentation: {augmentation} with probability {p_augment}")
    else:
        augmenter = None
    
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
                
                # Store fold performance across epochs
                fold_accuracies = []
                fold_precisions = []
                fold_recalls = []
                fold_f1_scores = []
                fold_auc_scores = []
                
                for epoch in range(num_epochs):
                    print(f"\nFold {fold_idx}, Epoch {epoch+1}/{num_epochs}")
                    
                    # Augment training data if requested, with epoch-awareness
                    if augmenter is not None:
                        epoch_train_data = create_augmented_data(
                            train_data, 
                            augmenter, 
                            method=augmentation,
                            augment_ratio=augment_ratio,
                            current_epoch=epoch
                        )
                    else:
                        epoch_train_data = train_data
                    
                    X_train = epoch_train_data['processed_message']
                    y_train = epoch_train_data['label']
                    X_test = test_data['processed_message']
                    y_test = test_data['label']
                    
                    # Create pipeline with TF-IDF and SVM
                    print(f"Training SVM model (Epoch {epoch+1})...")
                    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=10)),
                        ('svm', SVC(kernel='linear', C=1.0, probability=True))
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    macro_f1 = f1_score(y_test, y_pred, average='macro')
                    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Calculate AUC if applicable (binary classification)
                    try:
                        # Probability estimates for AUC
                        y_prob = pipeline.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_prob)
                    except:
                        # For multiclass or if probabilities can't be obtained
                        auc = 0.0
                        print("Warning: Could not calculate AUC score")
                    
                    # Save all metrics
                    fold_accuracies.append(accuracy)
                    fold_precisions.append(precision)
                    fold_recalls.append(recall)
                    fold_f1_scores.append(weighted_f1)
                    fold_auc_scores.append(auc)
                    
                    # Confusion matrix for this epoch
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                    plt.title(f'Confusion Matrix - Fold {fold_idx}, Epoch {epoch+1}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/confusion_matrix_fold_{fold_idx}_epoch_{epoch+1}.png")
                    plt.close()
                    
                    print(f"Epoch {epoch+1} results:")
                    print(f"  Accuracy:  {accuracy:.4f}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall:    {recall:.4f}")
                    print(f"  F1 Score:  {weighted_f1:.4f}")
                    print(f"  AUC Score: {auc:.4f}")
                
                # Find the best epoch based on F1 (you can change this criterion)
                best_epoch_idx = np.argmax(fold_f1_scores)
                best_epoch = best_epoch_idx + 1
                best_accuracy = fold_accuracies[best_epoch_idx]
                best_precision = fold_precisions[best_epoch_idx]
                best_recall = fold_recalls[best_epoch_idx]
                best_weighted_f1 = fold_f1_scores[best_epoch_idx]
                best_auc = fold_auc_scores[best_epoch_idx]
                
                # Plot all metrics across epochs
                plt.figure(figsize=(12, 8))
                plt.plot(range(1, num_epochs+1), fold_accuracies, marker='o', label='Accuracy')
                plt.plot(range(1, num_epochs+1), fold_precisions, marker='s', label='Precision')
                plt.plot(range(1, num_epochs+1), fold_recalls, marker='^', label='Recall')
                plt.plot(range(1, num_epochs+1), fold_f1_scores, marker='x', label='F1 Score')
                plt.plot(range(1, num_epochs+1), fold_auc_scores, marker='*', label='AUC')
                plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.title(f'Fold {fold_idx} - Performance Metrics Across Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/fold_{fold_idx}_metrics_performance.png")
                plt.close()
                
                # Store results for this fold
                results.append({
                    'fold': fold_idx,
                    'best_epoch': best_epoch,
                    'accuracy': best_accuracy,
                    'precision': best_precision,
                    'recall': best_recall,
                    'weighted_f1': best_weighted_f1,
                    'auc': best_auc,
                    'accuracies': fold_accuracies,
                    'precisions': fold_precisions,
                    'recalls': fold_recalls,
                    'f1_scores': fold_f1_scores,
                    'auc_scores': fold_auc_scores
                })
                
                print(f"Fold {fold_idx} best results (Epoch {best_epoch}):")
                print(f"  Accuracy:  {best_accuracy:.4f}")
                print(f"  Precision: {best_precision:.4f}")
                print(f"  Recall:    {best_recall:.4f}")
                print(f"  F1 Score:  {best_weighted_f1:.4f}")
                print(f"  AUC Score: {best_auc:.4f}")
                
        # ...existing code for handling fold_indices as a list...
    
    # Check if we have any results
    if not all_predictions:
        print("Error: No predictions were made. Check the fold indices and dataset.")
        return
    
    # Calculate final metrics on all predictions
    final_accuracy = accuracy_score(all_true_labels, all_predictions)
    final_precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    final_recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    final_macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    final_weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    # Calculate AUC for final results if possible
    try:
        y_binary = np.array(all_true_labels)
        y_pred_binary = np.array(all_predictions)
        final_auc = roc_auc_score(y_binary, y_pred_binary)
    except:
        final_auc = 0.0
        print("Warning: Could not calculate final AUC score")
    
    # Final report
    final_report = classification_report(all_true_labels, all_predictions, output_dict=True)
    
    # Add final results with all metrics
    results.append({
        'fold': 'Final',
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'macro_f1': final_macro_f1,
        'weighted_f1': final_weighted_f1,
        'auc': final_auc
    })
    
    # Save results to CSV, including all metrics
    results_df = pd.DataFrame([{k:v for k,v in r.items() if not k.endswith('s')} for r in results])
    results_df.to_csv(f"{output_dir}/svm_performance_results.csv", index=False)
    
    # Create a summary table with averages across folds (excluding the "Final" row)
    fold_results = [r for r in results if r['fold'] != 'Final']
    if fold_results:
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_precision = np.mean([r['precision'] for r in fold_results])
        avg_recall = np.mean([r['recall'] for r in fold_results])
        avg_weighted_f1 = np.mean([r['weighted_f1'] for r in fold_results])
        avg_auc = np.mean([r['auc'] for r in fold_results])
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
            'Average': [avg_accuracy, avg_precision, avg_recall, avg_weighted_f1, avg_auc],
            'Final': [final_accuracy, final_precision, final_recall, final_weighted_f1, final_auc]
        })
        summary_df.to_csv(f"{output_dir}/svm_summary_metrics.csv", index=False)
    
    # Save final detailed report
    final_report_df = pd.DataFrame(final_report).transpose()
    final_report_df.to_csv(f"{output_dir}/final_classification_report.csv")
    
    # Plot summary of all metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    values = [final_accuracy, final_precision, final_recall, final_weighted_f1, final_auc]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylim(0, 1.0)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Final Performance Metrics')
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_metrics_summary.png")
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nModel training and evaluation completed in {elapsed_time:.2f} seconds")
    print(f"\nFINAL METRICS SUMMARY:")
    print(f"{'='*50}")
    print(f"  Accuracy:  {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  F1 Score:  {final_weighted_f1:.4f}")
    print(f"  AUC Score: {final_auc:.4f}")
    print(f"{'='*50}")
    print(f"All results and reports saved to {output_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SVM model with data augmentation')
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'random', 'synonym', 'delete', 'swap', 'insert', 'backtranslation', 'cascaded'],
                        help='Type of text augmentation to use')
    parser.add_argument('--p_augment', type=float, default=0.5,
                        help='Probability of applying augmentation to a sample')
    parser.add_argument('--augment_ratio', type=float, default=1.0,
                        help='Ratio of training data to augment (1.0 = all samples)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train with different augmentations')
    
    args = parser.parse_args()
    
    main(
        augmentation=args.augmentation,
        p_augment=args.p_augment,
        augment_ratio=args.augment_ratio,
        num_epochs=args.num_epochs
    )
