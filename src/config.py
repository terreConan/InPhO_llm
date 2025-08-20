"""
Configuration and version tracking for InPhO LLM Research Project
"""

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    name: str
    provider: str
    model_id: str
    api_base: str
    max_tokens: int = 100
    temperature: float = 0.0
    cost_per_1k_tokens: float = 0.0  # USD per 1K tokens
    rate_limit_per_minute: int = 0
    
    def __post_init__(self):
        # Set default rate limits based on provider
        if self.provider == "groq":
            self.rate_limit_per_minute = 1000  # Groq's typical rate limit
        elif self.provider == "openai":
            self.rate_limit_per_minute = 60    # OpenAI's typical rate limit

# Current model configurations
MODELS = {
    "llama-3.1-8b-instant": ModelConfig(
        name="Llama 3.1 8B Instant",
        provider="groq",
        model_id="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        cost_per_1k_tokens=0.05,  # Groq pricing (approximate)
        rate_limit_per_minute=1000
    ),
    "gpt-4": ModelConfig(
        name="GPT-4",
        provider="openai", 
        model_id="gpt-4",
        api_base="https://api.openai.com/v1",
        cost_per_1k_tokens=0.03,  # OpenAI pricing (approximate)
        rate_limit_per_minute=60
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        provider="openai",
        model_id="gpt-3.5-turbo", 
        api_base="https://api.openai.com/v1",
        cost_per_1k_tokens=0.002,  # OpenAI pricing (approximate)
        rate_limit_per_minute=60
    ),
    "gemini-pro": ModelConfig(
        name="Gemini Pro",
        provider="google",
        model_id="gemini-pro",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        cost_per_1k_tokens=0.00125,  # Google pricing (approximate)
        rate_limit_per_minute=60
    ),
    "claude-3-sonnet": ModelConfig(
        name="Claude 3 Sonnet",
        provider="anthropic",
        model_id="claude-3-sonnet-20240229",
        api_base="https://api.anthropic.com",
        cost_per_1k_tokens=0.003,  # Anthropic pricing (input: $0.003, output: $0.015)
        rate_limit_per_minute=60
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider="anthropic", 
        model_id="claude-3-haiku-20240307",
        api_base="https://api.anthropic.com",
        cost_per_1k_tokens=0.00025,  # Anthropic pricing (input: $0.00025, output: $0.00125)
        rate_limit_per_minute=60
    )
}

# Version tracking
VERSION_INFO = {
    "project_version": "1.0.0",
    "sep_data_version": "2024-01-15",  # When SEP data was last updated
    "sep_data_source": "https://www.inphoproject.org/idea.json",
    "human_data_version": "2009-2010",  # Original human survey period
    "human_data_source": "idea_evaluation_with_all_user_info.csv",
    "last_processed": datetime.now().isoformat(),
    "python_version": "3.8+",
    "dependencies": {
        "requests": "2.31.0",
        "openai": "1.3.0", 
        "groq": "0.4.0",
        "pandas": "2.0.0",
        "matplotlib": "3.7.0",
        "seaborn": "0.12.0"
    }
}

# API Configuration
API_CONFIG = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY", 
        "base_url": "https://api.openai.com/v1",
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1.0
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1.0
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1.0
    }
}

import os

# Data file paths - flexible for different working directories
def get_data_paths():
    # Check if we're in the root directory or scripts directory
    if os.path.exists("data"):
        base_path = ""
    else:
        base_path = "../"
    
    return {
        "sep_edges": f"{base_path}data/raw/sep_idea_graph_edges.csv",
        "human_evaluations": f"{base_path}data/raw/idea_evaluation_with_all_user_info.csv", 
        "idea_mapping": f"{base_path}data/raw/idea_id_label_mapping.csv",
        "pairs": f"{base_path}data/raw/pairs.csv",
        "output_dir": f"{base_path}data/processed/"
    }

DATA_PATHS = get_data_paths()

# Expertise level mapping
EXPERTISE_LEVELS = {
    1: "amateur",
    2: "course_taker", 
    3: "phd_student",
    4: "expert"
}

def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    return MODELS.get(model_id)

def get_api_config(provider: str) -> Dict[str, Any]:
    """Get API configuration for a specific provider"""
    return API_CONFIG.get(provider, {})

def estimate_cost(model_id: str, num_tokens: int) -> float:
    """Estimate cost for a given number of tokens"""
    model = get_model_config(model_id)
    if model:
        return (num_tokens / 1000) * model.cost_per_1k_tokens
    return 0.0

def log_experiment(model_id: str, num_pairs: int, total_tokens: int, cost: float):
    """Log experiment details for reproducibility"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "model_id": model_id,
        "num_pairs": num_pairs,
        "total_tokens": total_tokens,
        "estimated_cost": cost,
        "version_info": VERSION_INFO
    }
    
    # Save to experiment log
    import json
    log_file = "experiment_log.json"
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Experiment logged: {model_id}, {num_pairs} pairs, ${cost:.4f} estimated cost") 