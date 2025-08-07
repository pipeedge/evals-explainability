"""
LLM Wrapper for Integration with Local and Remote LLMs

This module provides a unified interface for interacting with different LLM backends,
including local Ollama instances and remote APIs.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import from the existing local LLM setup
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    print("Warning: langchain_ollama not available. Please install to use local LLMs.")
    OllamaLLM = None


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    text: str
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class OllamaLLMWrapper(BaseLLM):
    """
    Wrapper for Ollama LLMs (local deployment)
    
    Integration with the existing Local_LLM_Samples.py setup
    """
    
    def __init__(self, base_url: str = "http://10.33.205.34:11112", 
                 model: str = "llama3.1:70b", timeout: int = 100):
        import requests
        
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to the Ollama server"""
        try:
            import requests
            
            payload = {
                "model": self.model,
                "prompt": "Hello",
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                print(f"✓ Connected to Ollama at {self.base_url} with model {self.model}")
            else:
                print(f"⚠ Warning: Ollama server returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Ollama server: {e}")
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the Ollama LLM with a prompt"""
        try:
            import requests
            
            start_time = time.time()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            # Add optional parameters
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                payload["top_p"] = kwargs["top_p"]
            if "max_tokens" in kwargs:
                payload["num_predict"] = kwargs["max_tokens"]
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                execution_time = time.time() - start_time
                return result.get("response", "No response generated")
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"Error invoking Ollama LLM: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "ollama",
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "type": "local"
        }


class OpenAILLMWrapper(BaseLLM):
    """
    Wrapper for OpenAI GPT models
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAILLMWrapper")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke OpenAI LLM with a prompt"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            execution_time = time.time() - start_time
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error invoking OpenAI LLM: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "openai",
            "model": self.model,
            "type": "remote"
        }


class AnthropicLLMWrapper(BaseLLM):
    """
    Wrapper for Anthropic Claude models
    """
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for AnthropicLLMWrapper")
        
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke Anthropic LLM with a prompt"""
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 2000),
                messages=[{"role": "user", "content": prompt}]
            )
            
            execution_time = time.time() - start_time
            return response.content[0].text
            
        except Exception as e:
            print(f"Error invoking Anthropic LLM: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "anthropic",
            "model": self.model,
            "type": "remote"
        }


class LLMWrapper:
    """
    Main LLM wrapper that provides a unified interface for different LLM backends
    
    This class manages different LLM implementations and provides additional features
    like response caching, retry logic, and performance monitoring.
    """
    
    def __init__(self, llm_backend: BaseLLM, enable_caching: bool = True):
        self.llm_backend = llm_backend
        self.enable_caching = enable_caching
        self.response_cache = {} if enable_caching else None
        self.performance_stats = {
            "total_calls": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "errors": 0
        }
    
    def invoke(self, prompt: str, use_cache: bool = True, **kwargs) -> str:
        """
        Invoke the LLM with enhanced features
        
        Args:
            prompt: Input prompt for the LLM
            use_cache: Whether to use response caching
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            LLM response text
        """
        start_time = time.time()
        self.performance_stats["total_calls"] += 1
        
        # Check cache if enabled
        cache_key = self._generate_cache_key(prompt, kwargs)
        if use_cache and self.enable_caching and cache_key in self.response_cache:
            self.performance_stats["cache_hits"] += 1
            return self.response_cache[cache_key]
        
        try:
            # Invoke the LLM backend
            response = self.llm_backend.invoke(prompt, **kwargs)
            
            # Cache the response if enabled
            if use_cache and self.enable_caching:
                self.response_cache[cache_key] = response
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats["total_time"] += execution_time
            
            return response
            
        except Exception as e:
            self.performance_stats["errors"] += 1
            print(f"Error in LLM invocation: {e}")
            return f"Error: {str(e)}"
    
    def invoke_with_retry(self, prompt: str, max_retries: int = 3, 
                         retry_delay: float = 1.0, **kwargs) -> str:
        """
        Invoke LLM with retry logic for robust operation
        
        Args:
            prompt: Input prompt for the LLM
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            LLM response text
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.invoke(prompt, **kwargs)
                if not response.startswith("Error:"):
                    return response
                    
            except Exception as e:
                if attempt == max_retries:
                    return f"Failed after {max_retries} retries: {str(e)}"
                
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return "Maximum retries exceeded"
    
    def batch_invoke(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Invoke LLM with multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            List of LLM responses
        """
        responses = []
        for prompt in prompts:
            response = self.invoke(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        if stats["total_calls"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_calls"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_calls"]
            stats["error_rate"] = stats["errors"] / stats["total_calls"]
        
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the underlying LLM"""
        return self.llm_backend.get_model_info()
    
    def clear_cache(self):
        """Clear the response cache"""
        if self.response_cache:
            self.response_cache.clear()
            print("Response cache cleared")
    
    def _generate_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key for the prompt and parameters"""
        import hashlib
        
        # Create a string representation of prompt + kwargs
        cache_input = f"{prompt}_{json.dumps(kwargs, sort_keys=True)}"
        
        # Generate hash
        cache_key = hashlib.md5(cache_input.encode()).hexdigest()
        
        return cache_key


def create_llm_wrapper(llm_type: str = "ollama", **kwargs) -> LLMWrapper:
    """
    Factory function to create LLM wrapper with different backends
    
    Args:
        llm_type: Type of LLM ("ollama", "openai", "anthropic")
        **kwargs: Additional arguments for the specific LLM backend
        
    Returns:
        Configured LLMWrapper instance
    """
    if llm_type.lower() == "ollama":
        backend = OllamaLLMWrapper(**kwargs)
    elif llm_type.lower() == "openai":
        backend = OpenAILLMWrapper(**kwargs)
    elif llm_type.lower() == "anthropic":
        backend = AnthropicLLMWrapper(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    return LLMWrapper(backend)


# Convenience function to create default wrapper using the same config as Local_LLM_Samples.py
def create_default_llm_wrapper() -> LLMWrapper:
    """
    Create default LLM wrapper using the same configuration as Local_LLM_Samples.py
    """
    return create_llm_wrapper(
        llm_type="ollama",
        base_url="http://10.33.205.34:11112",
        model="llama3.1:70b",
        timeout=100
    ) 