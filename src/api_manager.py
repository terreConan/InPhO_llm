"""
API Manager for InPhO LLM Research Project
Handles multiple LLM providers, cost tracking, caching, and rate limiting
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import requests

from config import get_model_config, get_api_config, estimate_cost, log_experiment

class APIManager:
    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        self.model_id = model_id
        self.model_config = get_model_config(model_id)
        self.api_config = get_api_config(self.model_config.provider)
        
        # Initialize API client
        self.client = self._initialize_client()
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        
        # Cache for API responses
        self.cache_file = f"api_cache_{model_id.replace('-', '_')}.json"
        self.cache = self._load_cache()
        
        # Rate limiting
        self.last_call_time = 0
        self.calls_this_minute = 0
        self.minute_start = time.time()
    
    def _initialize_client(self):
        """Initialize the appropriate API client"""
        api_key = os.getenv(self.api_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {self.model_config.provider}")
        
        if self.model_config.provider == "groq":
            from groq import Groq
            return Groq(api_key=api_key)
        elif self.model_config.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        elif self.model_config.provider == "google":
            # Google API setup would go here
            raise NotImplementedError("Google API not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load API response cache"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """Save API response cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt"""
        return hashlib.md5(f"{self.model_id}:{prompt}".encode()).hexdigest()
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.minute_start >= 60:
            self.calls_this_minute = 0
            self.minute_start = current_time
        
        # Check if we're at the rate limit
        if self.calls_this_minute >= self.model_config.rate_limit_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                print(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.calls_this_minute = 0
                self.minute_start = time.time()
        
        # Ensure minimum delay between calls
        time_since_last = current_time - self.last_call_time
        min_delay = 60.0 / self.model_config.rate_limit_per_minute
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)
        
        self.last_call_time = time.time()
        self.calls_this_minute += 1
    
    def call_api(self, prompt: str, system_prompt: str = None, use_cache: bool = True) -> Tuple[str, int, float]:
        """
        Make API call with caching and cost tracking
        
        Returns:
            Tuple of (response_text, tokens_used, cost)
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        if use_cache and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            print(f"Using cached response for prompt: {prompt[:50]}...")
            return cached_response["text"], cached_response["tokens"], cached_response["cost"]
        
        # Rate limiting
        self._rate_limit()
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Make API call with retry logic
        max_retries = self.api_config["max_retries"]
        retry_delay = self.api_config["retry_delay"]
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.model_id,
                    messages=messages,
                    max_tokens=self.model_config.max_tokens,
                    temperature=self.model_config.temperature
                )
                
                # Extract response
                response_text = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens
                cost = estimate_cost(self.model_id, tokens_used)
                
                # Update tracking
                self.total_tokens += tokens_used
                self.total_cost += cost
                self.api_calls += 1
                
                # Cache the response
                if use_cache:
                    self.cache[cache_key] = {
                        "text": response_text,
                        "tokens": tokens_used,
                        "cost": cost,
                        "timestamp": datetime.now().isoformat()
                    }
                    self._save_cache()
                
                print(f"API call successful: {tokens_used} tokens, ${cost:.4f}")
                return response_text, tokens_used, cost
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"API call failed after {max_retries + 1} attempts: {e}")
                    raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "model_id": self.model_id,
            "total_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cache_hits": len(self.cache),
            "average_cost_per_call": self.total_cost / max(self.api_calls, 1)
        }
    
    def estimate_total_cost(self, num_pairs: int) -> float:
        """Estimate total cost for processing a number of idea pairs"""
        # Estimate tokens per pair (system prompt + user prompt + response)
        estimated_tokens_per_pair = 200  # Conservative estimate
        total_tokens = num_pairs * estimated_tokens_per_pair
        return estimate_cost(self.model_id, total_tokens)
    
    def log_experiment(self, num_pairs: int):
        """Log experiment details"""
        log_experiment(self.model_id, num_pairs, self.total_tokens, self.total_cost)

# Example usage
if __name__ == "__main__":
    # Test with a small number of pairs
    api_manager = APIManager("llama-3.1-8b-instant")
    
    test_prompt = "How related is ethics to virtue?"
    system_prompt = """You're a philosophy researcher familiar with the Stanford Encyclopedia of Philosophy.
    For each pair of ideas, please answer:
    1) "How related is <Idea A> to <Idea B>?"
    — Not Related / Marginally Related / Somewhat Related / Related / Highly Related
    2) If your answer is not "Not Related," say whether A is More Specific Than / More General Than / As General As / Incomparable To.
    For both parts, say the answer only concisely."""
    
    try:
        response, tokens, cost = api_manager.call_api(test_prompt, system_prompt)
        print(f"Response: {response}")
        print(f"Usage stats: {api_manager.get_usage_stats()}")
    except Exception as e:
        print(f"Error: {e}") 