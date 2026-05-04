"""
Multi-Provider Embedding Service.

Supports Ollama, OpenAI, and Google embedding backends.
"""

import logging
import os
import asyncio
from typing import List, Optional
from enum import Enum

import aiohttp

from agent_reminiscence.config import Config

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    GOOGLE = "google"


class EmbeddingService:
    """
    Multi-provider service for generating text embeddings.

    Supports Ollama (local), OpenAI, and Google embedding backends.
    """

    def __init__(self, config: Config):
        """
        Initialize embedding service.

        Args:
            config: Configuration object with embedding settings
        """
        self.config = config
        self.provider = self._resolve_provider(config.embedding_provider)
        self.vector_dimension = config.vector_dimension

        # Provider-specific setup
        if self.provider == EmbeddingProvider.OLLAMA:
            self.base_url = config.ollama_base_url.rstrip("/v1").rstrip("/")
            self.model = config.embedding_model
        elif self.provider == EmbeddingProvider.OPENAI:
            self.model = config.openai_embedding_model
            self._openai_api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        elif self.provider == EmbeddingProvider.GOOGLE:
            self.model = config.google_embedding_model
            self._google_api_key = config.google_api_key or os.getenv("GOOGLE_API_KEY")

        logger.info(f"Embedding service initialized: provider={self.provider.value}, model={self.model}")

    def _resolve_provider(self, provider: str) -> EmbeddingProvider:
        """Resolve and validate provider string to enum."""
        normalized = provider.lower().strip()
        try:
            return EmbeddingProvider(normalized)
        except ValueError:
            logger.warning(
                f"Unknown embedding provider '{provider}', defaulting to ollama. "
                f"Valid options: ollama, openai, google"
            )
            return EmbeddingProvider.OLLAMA

    async def get_embedding(self, text: str, timeout: int = 30) -> List[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed
            timeout: Request timeout in seconds (default: 30)

        Returns:
            List of float values representing the text embedding

        Raises:
            RuntimeError: If embedding generation fails

        Example:
            embedding = await service.get_embedding("Hello world")
            print(f"Embedding dimension: {len(embedding)}")
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.vector_dimension

        if self.provider == EmbeddingProvider.OLLAMA:
            return await self._get_embedding_ollama(text, timeout)
        elif self.provider == EmbeddingProvider.OPENAI:
            return await self._get_embedding_openai(text, timeout)
        elif self.provider == EmbeddingProvider.GOOGLE:
            return await self._get_embedding_google(text, timeout)

    async def _get_embedding_ollama(self, text: str, timeout: int) -> List[float]:
        """Generate embedding using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": text,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    embedding = result.get("embedding", [])

                    if not embedding:
                        raise ValueError("No embedding returned from Ollama API")

                    if len(embedding) != self.vector_dimension:
                        logger.warning(
                            f"Expected embedding dimension {self.vector_dimension}, "
                            f"got {len(embedding)}. Check model configuration."
                        )

                    return embedding

        except Exception as e:
            logger.error(f"Error generating embedding via Ollama at {self.base_url}: {e}")
            return [0.0] * self.vector_dimension

    async def _get_embedding_openai(self, text: str, timeout: int) -> List[float]:
        """Generate embedding using OpenAI API."""
        if not self._openai_api_key:
            logger.error("OpenAI API key not configured")
            return [0.0] * self.vector_dimension

        try:
            payload = {
                "input": text,
                "model": self.model,
            }

            headers = {
                "Authorization": f"Bearer {self._openai_api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    embedding = result["data"][0]["embedding"]

                    if len(embedding) != self.vector_dimension:
                        logger.warning(
                            f"Expected embedding dimension {self.vector_dimension}, "
                            f"got {len(embedding)}. Check model configuration."
                        )

                    return embedding

        except aiohttp.ClientError as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return [0.0] * self.vector_dimension

    async def _get_embedding_google(self, text: str, timeout: int) -> List[float]:
        """Generate embedding using Google Generative AI API."""
        if not self._google_api_key:
            logger.error("Google API key not configured")
            return [0.0] * self.vector_dimension

        try:
            payload = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
            }

            headers = {
                "x-goog-api-key": self._google_api_key,
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    embedding = result["embedding"]["values"]

                    if len(embedding) != self.vector_dimension:
                        logger.warning(
                            f"Expected embedding dimension {self.vector_dimension}, "
                            f"got {len(embedding)}. Check model configuration."
                        )

                    return embedding

        except aiohttp.ClientError as e:
            logger.error(f"Error calling Google API: {e}")
            return [0.0] * self.vector_dimension

    async def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        timeout: int = 30,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of concurrent requests (default: 10)
            timeout: Request timeout per embedding (default: 30)

        Returns:
            List of embeddings in the same order as input texts

        Example:
            texts = ["Hello", "World", "Test"]
            embeddings = await service.get_embeddings_batch(texts)
            print(f"Generated {len(embeddings)} embeddings")
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Process batch concurrently
            batch_embeddings = await asyncio.gather(
                *[self.get_embedding(text, timeout) for text in batch],
                return_exceptions=True,
            )

            # Handle any exceptions in the batch
            for j, result in enumerate(batch_embeddings):
                if isinstance(result, Exception):
                    logger.error(f"Error generating embedding for text {i+j}: {result}")
                    embeddings.append([0.0] * self.vector_dimension)
                else:
                    embeddings.append(result)

            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings

    async def verify_connection(self) -> bool:
        """
        Verify that embedding provider is accessible.

        Returns:
            True if provider is available and working, False otherwise
        """
        if self.provider == EmbeddingProvider.OLLAMA:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        models = result.get("models", [])
                        model_names = [m.get("name", "") for m in models]

                        if self.model in model_names:
                            logger.info(f"Ollama connection verified, model {self.model} available")
                            return True
                        else:
                            logger.warning(
                                f"Ollama is running but model {self.model} not found. "
                                f"Available models: {model_names}"
                            )
                            return False

            except Exception as e:
                logger.error(f"Failed to verify Ollama connection: {e}")
                return False

        elif self.provider == EmbeddingProvider.OPENAI:
            if not self._openai_api_key:
                logger.error("OpenAI API key not configured")
                return False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {self._openai_api_key}"},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status == 200:
                            logger.info("OpenAI API connection verified")
                            return True
                        logger.error(f"OpenAI API returned status {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Failed to verify OpenAI connection: {e}")
                return False

        elif self.provider == EmbeddingProvider.GOOGLE:
            if not self._google_api_key:
                logger.error("Google API key not configured")
                return False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}",
                        params={"key": self._google_api_key},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status == 200:
                            logger.info("Google API connection verified")
                            return True
                        logger.error(f"Google API returned status {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Failed to verify Google connection: {e}")
                return False

        return False

    async def ensure_model_available(self) -> bool:
        """
        Ensure the embedding model is available by attempting to generate a test embedding.

        Returns:
            True if model is available and working, False otherwise
        """
        try:
            test_embedding = await self.get_embedding("test", timeout=10)

            # Check if we got a real embedding (not zero vector)
            if sum(test_embedding) == 0:
                logger.error(f"Model {self.model} returned zero vector")
                return False

            logger.info(f"Model {self.model} is available and working")
            return True

        except Exception as e:
            logger.error(f"Model {self.model} is not available: {e}")
            return False


