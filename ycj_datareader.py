import pandas as pd
import numpy as np
import re
import os
import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))

class YouTubeCommentDataset(Dataset):
    
    def __init__(self, 
                 file_path="./data/youtube_chat_jogja_clean.csv", 
                 fold=0,
                 n_folds=5,
                 split="train",
                 max_length=128,
                 tokenizer_name="indobenchmark/indobert-base-p1",
                 apply_preprocessing=True,
                 random_state=42,
                 folds_file="ycj_split.json"):
        """
        Initialize the YouTube Comment Dataset for spam detection
        
        Args:
            file_path: Path to the CSV file containing comments
            fold: Which fold to use (0 to n_folds-1)
            n_folds: Number of folds for cross-validation
            split: 'train' or 'val'
            max_length: Maximum length for tokenized inputs
            tokenizer_name: Pre-trained tokenizer to use
            apply_preprocessing: Whether to apply text preprocessing
            random_state: Random seed for reproducibility
            folds_file: Path to save/load the fold indices
        """
        self.file_path = file_path
        self.fold = fold
        self.n_folds = n_folds
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.apply_preprocessing = apply_preprocessing
        self.random_state = random_state
        self.folds_file = folds_file
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load and process data
        print(f"Loading data from {file_path}...")
        self.load_data()
        
        # Create or load folds
        self.setup_folds()
        
        # Set up indices for this fold and split
        self.setup_indices()
        
        print(f"Created {split} dataset for fold {fold+1}/{n_folds} with {len(self.indices)} samples")
    
    def load_data(self):
        """Load data from CSV file and preprocess text if needed"""
        # Load data
        self.df = pd.read_csv(self.file_path)
        
        # Clean data
        self.df = self.df.dropna(subset=['message', 'label'])
        self.df['label'] = self.df['label'].astype(int)
        
        print(f"Loaded {len(self.df)} comments")
    
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, URLs, and stopwords using NLTK"""
        if not isinstance(text, str):
            return ""
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Step 3: Remove special characters but keep Indonesian characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Step 4: Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 5: Tokenize text using NLTK's word_tokenize
        words = nltk.word_tokenize(text)
        
        # Step 6: Remove stopwords using the INDONESIAN_STOPWORDS set
        words = [word for word in words if word not in INDONESIAN_STOPWORDS and len(word) > 1]
        
        # Step 7: Join words back together
        return ' '.join(words)
    
    def setup_folds(self):
        """Create or load stratified folds"""
        if os.path.exists(self.folds_file):
            self.load_folds()
        else:
            self.create_folds()
    
    def create_folds(self):
        """Create stratified k-fold indices and save to file"""
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Generate train/val indices for cross-validation
        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['label'])):
            fold_indices[f"fold_{fold}"] = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            }
        
        # Save to file
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices': fold_indices,
                'created_at': pd.Timestamp.now().isoformat(),
                'n_samples': len(self.df),
                'n_folds': self.n_folds,
                'random_state': self.random_state
            }, f)
            
        self.fold_indices = fold_indices
        print(f"Created {self.n_folds}-fold indices and saved to {self.folds_file}")
    
    def load_folds(self):
        """Load fold indices from file"""
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
        
        self.fold_indices = fold_data['fold_indices']
        print(f"Loaded fold indices from {self.folds_file}")
        
        # Verify compatibility
        if len(self.df) != fold_data['n_samples']:
            print(f"Warning: Dataset size ({len(self.df)}) differs from when folds were created ({fold_data['n_samples']})")
            print("Creating new folds...")
            os.remove(self.folds_file)
            self.create_folds()
    
    def setup_indices(self):
        """Set up indices for the current fold and split"""
        fold_key = f"fold_{self.fold}"
        if self.split == "train":
            self.indices = self.fold_indices[fold_key]['train_indices']
        else:  # val
            self.indices = self.fold_indices[fold_key]['val_indices']
    
    def get_class_weights(self):
        """Calculate class weights to handle imbalanced classes"""
        class_counts = self.df['label'].value_counts().sort_index()
        weights = torch.FloatTensor(
            [len(self.df) / (len(class_counts) * count) for count in class_counts]
        )
        return weights
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset with detailed step-by-step processing"""
        # Step 1: Get the index from our indices list
        original_idx = self.indices[idx]
        
        # Step 2: Extract the comment and label
        original_message = str(self.df.iloc[original_idx]['message'])
        label = int(self.df.iloc[original_idx]['label'])
        
        # Apply preprocessing if specified
        if self.apply_preprocessing:
            message = self.preprocess_text(original_message)
        else:
            message = original_message
        
        # Step 3: Tokenize the comment
        encoding = self.tokenizer(
            message,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Step 4: Prepare the final output dictionary
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'original_text': original_message,  # Original unprocessed text
            'processed_text': message,  # Processed or original text based on self.apply_preprocessing
            'original_label': label  # Original label (0 or 1)
        }
        
        return sample


# Example usage
if __name__ == "__main__":
    # Create training dataset for fold 0
    train_dataset = YouTubeCommentDataset(
        file_path="./data/youtube_chat_jogja_clean.csv",
        fold=0,
        split="train",
        apply_preprocessing=True
    )
    
    # Create validation dataset for fold 0
    val_dataset = YouTubeCommentDataset(
        file_path="./data/youtube_chat_jogja_clean.csv",
        fold=0,
        split="val",
        apply_preprocessing=True
    )
    
    # Print dataset stats
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Get a sample
    sample = train_dataset[1]
    print("\nSample from dataset:")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Original text: {sample['original_text'][:100]}...")
    print(f"Processed text: {sample['processed_text'][:100]}...")
    print(f"Label: {sample['original_label']} (0=normal, 1=spam)")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    
    # Calculate class weights
    weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    # Print class distribution
    class_distribution = train_dataset.df['label'].value_counts()
    print(f"\nClass distribution:")
    print(f"Normal (0): {class_distribution.get(0, 0)}")
    print(f"Spam (1): {class_distribution.get(1, 0)}")