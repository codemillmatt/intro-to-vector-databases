"""
Shared embedding utilities for the vector database demos.
This module provides a consistent interface for generating embeddings
using either Ollama (local) or other providers.
"""

import os
from typing import Optional

# Try to import ollama, but fall back gracefully
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import sentence-transformers for offline/fallback use
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingClient:
    """
    A client for generating text embeddings.
    
    Supports multiple backends:
    - Ollama (local, requires running Ollama server)
    - sentence-transformers (offline, local)
    """
    
    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        ollama_host: Optional[str] = None
    ):
        """
        Initialize the embedding client.
        
        Args:
            provider: The embedding provider to use. Options: "ollama", "sentence-transformers", "auto"
            model: The model name to use. Defaults depend on provider.
            ollama_host: The Ollama server host (e.g., "http://localhost:11434")
        """
        self.provider = self._select_provider(provider)
        self.model = model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._st_model = None
        
        # Set default models based on provider
        if self.model is None:
            if self.provider == "ollama":
                self.model = "all-minilm"
            elif self.provider == "sentence-transformers":
                self.model = "all-MiniLM-L6-v2"
    
    def _select_provider(self, provider: str) -> str:
        """Select the best available provider."""
        if provider != "auto":
            return provider
        
        # Auto-select based on availability
        if OLLAMA_AVAILABLE:
            return "ollama"
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            return "sentence-transformers"
        else:
            raise RuntimeError(
                "No embedding provider available. "
                "Install either 'ollama' or 'sentence-transformers'."
            )
    
    def _get_st_model(self):
        """Lazy load sentence-transformers model."""
        if self._st_model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("sentence-transformers is not installed")
            self._st_model = SentenceTransformer(self.model)
        return self._st_model
    
    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        if self.provider == "ollama":
            return self._embed_ollama(texts)
        elif self.provider == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package is not installed")
        
        embeddings = []
        for text in texts:
            # Try the newer 'input' parameter first, fall back to 'prompt'
            try:
                response = ollama.embed(model=self.model, input=text)
            except TypeError:
                # Fallback for older ollama versions using 'prompt'
                response = ollama.embeddings(model=self.model, prompt=text)
            
            # Handle both old and new ollama response formats
            if hasattr(response, 'embeddings'):
                embeddings.append(response.embeddings[0])
            elif isinstance(response, dict) and 'embeddings' in response:
                embeddings.append(response['embeddings'][0])
            elif isinstance(response, dict) and 'embedding' in response:
                embeddings.append(response['embedding'])
            else:
                embeddings.append(response)
        return embeddings
    
    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        model = self._get_st_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        # Common model dimensions
        dim_map = {
            # Ollama models
            "all-minilm": 384,
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            # Sentence-transformers models
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
        }
        return dim_map.get(self.model, 384)


def get_embedding_client(**kwargs) -> EmbeddingClient:
    """
    Factory function to create an embedding client.
    
    This is the recommended way to get an embedding client,
    as it handles configuration from environment variables.
    """
    return EmbeddingClient(**kwargs)
