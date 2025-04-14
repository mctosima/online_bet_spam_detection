#!/bin/bash

# Set up variables
GITHUB_USERNAME="mctosima"
GITHUB_TOKEN="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"  # Replace with your actual token
REPO_NAME="online_bet_spam_detection"
REPO_URL="https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"
TEMP_DIR="temp_repo"
WANDB_API_KEY="48c4d7778a27bdc532d7020d8ea71a12283d00c9"  # Replace with your actual W&B API key

echo "=== Starting Environment Setup ==="

# Install uv
echo "Installing uv package manager..."
pip install uv

# Create virtual environment using uv with Python 3.10
echo "Creating virtual environment with Python 3.10..."
uv venv --python=python3.10
source .venv/bin/activate

# Install required packages
echo "Installing required Python packages..."
uv pip install numpy pandas matplotlib scikit-learn nltk wandb transformers openpyxl pytz torch

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

# Clone the repository to a temporary directory
echo "Cloning repository from GitHub..."
git clone $REPO_URL $TEMP_DIR

# Use rsync to copy all contents preserving file attributes
# The trailing slash on source ensures contents are copied directly
echo "Copying repository contents to current directory..."
cp -a $TEMP_DIR/. ./

# Remove the temporary clone directory
echo "Cleaning up..."
rm -rf $TEMP_DIR

echo "=== Setup Complete ==="
echo "Your environment is ready with:"
echo "- Python virtual environment (activate with source .venv/bin/activate)"
echo "- Required packages installed"
echo "- Repository contents in current directory"
echo "- Weights & Biases configured"
echo ""
echo "To run training with W&B logging, use:"
echo "python ycj_train.py --use_wandb --wandb_project YOUR_PROJECT_NAME --wandb_entity YOUR_WANDB_USERNAME"

# Activate the virtual environment again to ensure it's active
source .venv/bin/activate

# Ensure temp_repo is removed completely
if [ -d "$TEMP_DIR" ]; then
    echo "Removing temporary repository directory..."
    rm -rf "$TEMP_DIR"
    echo "Temporary directory removed successfully."
else
    echo "Temporary directory not found. Already cleaned up."
fi

# Activate the virtual environment again to ensure it's active
source .venv/bin/activate