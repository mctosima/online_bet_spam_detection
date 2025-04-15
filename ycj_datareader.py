import pandas as pd
import numpy as np
import re
import os
import json
import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

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
    
    def simulate_backtranslation(self, words, error_prob=0.2):
        """
        Simulate back-translation by introducing minor changes
        that mimic translation artifacts
        """
        if not words:
            return words
            
        new_words = words.copy()
        
        # Possible operations:
        # 1. Remove articles (a, an, the)
        # 2. Change word order slightly
        # 3. Replace some words with synonyms
        
        # Remove articles (simple for demonstration)
        articles = {'a', 'an', 'the', 'yang', 'ini', 'itu'}
        new_words = [w for w in new_words if w.lower() not in articles]
        
        # Change word order (simple swap) - 20% chance
        if random.random() < 0.2 and len(new_words) > 3:
            # Find a sub-segment to swap
            start_idx = random.randint(0, len(new_words) - 3)
            segment_len = min(3, len(new_words) - start_idx)
            segment = new_words[start_idx:start_idx+segment_len]
            # Shuffle the segment
            random.shuffle(segment)
            # Replace in original
            new_words[start_idx:start_idx+segment_len] = segment
        
        # Replace some words with synonyms
        for i, word in enumerate(new_words):
            if random.random() < error_prob and word not in self.stopwords:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    new_words[i] = random.choice(synonyms)
        
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
                    'insert', 'backtranslation', 'cascaded', or 'none')
        
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
            method = random.choice(['synonym', 'delete', 'swap', 'insert', 'backtranslation', 'cascaded'])
        
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
        elif method == 'backtranslation':
            # Simulate back-translation
            words = self.simulate_backtranslation(words)
            return ' '.join(words)
        elif method == 'cascaded':
            # Apply cascaded augmentation (multiple techniques in sequence)
            return self.cascaded_augmentation(text)
        
        # Default return (shouldn't reach here)
        return text

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
                 folds_file="ycj_split.json",
                 use_local_tokenizer=True,
                 augmentation='none',
                 p_augment=0.5,
                 current_epoch=0):
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
            use_local_tokenizer: Whether to use local tokenizer or download from huggingface
            augmentation: Type of text augmentation to apply ('none', 'random', 'synonym', 'delete', 'swap', 'insert', 'backtranslation')
            p_augment: Probability of applying augmentation to a sample
            current_epoch: Current training epoch (used for dynamic augmentation)
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
        self.augmentation = augmentation
        self.p_augment = p_augment
        self.current_epoch = current_epoch
        
        # Initialize augmenter - only for training split
        if split == "train" and augmentation != 'none':
            self.augmenter = TextAugmenter(p_augment=p_augment, language='indonesian')
        else:
            self.augmenter = None
        
        # Initialize tokenizer
        if use_local_tokenizer and os.path.exists("./indobert-base-p1"):
            print("Using local tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("./indobert-base-p1")
        else:
            print(f"Loading tokenizer from {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load and process data
        print(f"Loading data from {file_path}...")
        self.load_data()
        
        # Create or load folds
        self.setup_folds()
        
        # Set up indices for this fold and split
        self.setup_indices()
        
        print(f"Created {split} dataset for fold {fold+1}/{n_folds} with {len(self.indices)} samples")
        if split == "train" and augmentation != 'none':
            print(f"Using text augmentation: {augmentation} with probability {p_augment}")
    
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
    
    def set_epoch(self, epoch):
        """
        Set the current epoch for epoch-aware augmentation
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        # Reset random seed based on epoch to ensure different augmentations
        random.seed(self.random_state + epoch)
    
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
        
        # Apply augmentation if we're in training split and augmentation is enabled
        if self.split == 'train' and self.augmenter is not None:
            # Use a deterministic but different seed for each epoch and each sample
            sample_epoch_seed = hash(str(original_idx) + str(self.current_epoch)) % 10000
            random.seed(sample_epoch_seed)
            
            # Choose augmentation method dynamically based on epoch if set to 'random'
            if self.augmentation == 'random':
                # Change available methods based on epoch to create diversity
                epoch_mod = self.current_epoch % 5
                if epoch_mod == 0:
                    aug_method = random.choice(['synonym', 'delete'])
                elif epoch_mod == 1:
                    aug_method = random.choice(['swap', 'insert'])
                elif epoch_mod == 2:
                    aug_method = random.choice(['backtranslation', 'synonym'])
                elif epoch_mod == 3:
                    aug_method = random.choice(['delete', 'swap'])
                else:
                    aug_method = random.choice(['synonym', 'delete', 'swap', 'insert', 'backtranslation'])
            else:
                aug_method = self.augmentation
                
            # Apply the selected augmentation
            augmented_message = self.augmenter.augment(message, method=aug_method)
            
            # Only use augmented text if it's not empty
            if augmented_message.strip():
                message = augmented_message
                
            # Reset random seed
            random.seed()
        
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