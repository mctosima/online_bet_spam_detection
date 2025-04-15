#!/bin/bash

# Set up variables
GITHUB_USERNAME="mctosima"
GITHUB_TOKEN="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"  # Replace with your actual token
REPO_NAME="online_bet_spam_detection"
REPO_URL="https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "=== Starting Git Repository Update ==="

# Check if the directory is a git repository
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
    echo "Would you like to clone the repository? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        # Clone the repository directly to the current directory
        echo "Cloning repository from GitHub..."
        git clone $REPO_URL .
        
        if [ $? -eq 0 ]; then
            echo "Repository successfully cloned."
        else
            echo "Error: Failed to clone the repository."
            exit 1
        fi
    else
        echo "Operation cancelled."
        exit 0
    fi
fi

echo "=== Repository Update Complete ==="
echo "Your repository is now up-to-date with the remote."

# Activate the virtual environment if it exists
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Note: No virtual environment found. If needed, run ps_setup.sh to set up the environment."
fi
