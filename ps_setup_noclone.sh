#!/bin/bash

# Set up variables
GITHUB_USERNAME="mctosima"
GITHUB_TOKEN="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"  # Replace with your actual token
REPO_NAME="online_bet_spam_detection"
REPO_URL="https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"
WANDB_API_KEY="48c4d7778a27bdc532d7020d8ea71a12283d00c9"  # Replace with your actual W&B API key

echo "=== Starting Environment Setup ==="

# Repository update logic
echo "=== Updating Repository ==="
if [ -d ".git" ]; then
    echo "Git repository found in current directory."
    
    # Ensure we're using the correct remote URL with authentication
    echo "Updating remote URL with authentication..."
    git remote set-url origin $REPO_URL
    
    # Fetch and pull the latest changes
    echo "Fetching the latest changes..."
    git fetch origin
    
    echo "Pulling the latest changes..."
    if git pull origin $(git rev-parse --abbrev-ref HEAD); then
        echo "Successfully pulled the latest changes."
    else
        echo "Error: Failed to pull the latest changes. There might be conflicts."
        exit 1
    fi
else
    echo "No Git repository found in the current directory."
    echo "Cloning repository from GitHub..."
    git clone $REPO_URL .
    
    if [ $? -eq 0 ]; then
        echo "Repository successfully cloned."
    else
        echo "Error: Failed to clone the repository."
        exit 1
    fi
fi

# Environment setup logic
echo "=== Setting Up Python Environment ==="

# Install uv
echo "Installing uv package manager..."
pip install uv

# Create virtual environment using uv with Python 3.10
echo "Creating virtual environment with Python 3.10..."
uv venv --python=python3.10
source .venv/bin/activate

# Install required packages (with specific versions optimized for GPU)
echo "Installing required Python packages..."
uv pip install numpy pandas matplotlib scikit-learn seaborn nltk wandb transformers openpyxl pytz torch torchinfo

# Set up Weights & Biases
echo "Setting up Weights & Biases..."
if [ -z "$WANDB_API_KEY" ] || [ "$WANDB_API_KEY" == "YOUR_WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set properly in script. Please log in manually:"
    wandb login
else
    echo "Logging into Weights & Biases..."
    wandb login $WANDB_API_KEY
fi

# Create a wandb config directory if it doesn't exist
mkdir -p ~/.config/wandb

echo "=== Setup Complete ==="
echo "Your GPU environment is ready with:"
echo "- Python virtual environment (activate with source .venv/bin/activate)"
echo "- Required packages installed with GPU support"
echo "- Repository updated to the latest version"
echo "- Weights & Biases configured"
echo ""
echo "To run training with W&B logging, use:"
echo "python ycj_train.py --use_wandb --wandb_project YOUR_PROJECT_NAME --wandb_entity YOUR_WANDB_USERNAME --device cuda"

# Activate the virtual environment again to ensure it's active
source .venv/bin/activate
