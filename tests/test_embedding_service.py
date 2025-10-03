"""
Tests for Embedding Service (services/embedding.py).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_mem.config.settings import Config
from agent_mem.services.embedding import EmbeddingService


class TestEmbeddingService:
    """Test EmbeddingService class."""

    def test_initialization(self, test_config: Config):
        """Test service initialization."""
        service = EmbeddingService(test_config)
        assert service.config == test_config
        assert service.base_url == test_config.ollama_base_url
        assert service.model == test_config.embedding_model

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_config: Config):
        """Test successful embedding generation."""
        service = EmbeddingService(mock_config)

        mock_response = {"embedding": [0.1, 0.2, 0.3, 0.4]}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test text")

            assert result == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_get_embedding_fallback(self, mock_config: Config):
        """Test embedding generation fallback on error."""
        service = EmbeddingService(mock_config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.post = MagicMock(side_effect=Exception("Connection error"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test text")

            # Should return zero vector as fallback
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self, mock_config: Config):
        """Test batch embedding generation."""
        service = EmbeddingService(mock_config)

        texts = ["text 1", "text 2", "text 3"]

        # Mock individual embedding calls
        service.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        results = await service.get_embeddings_batch(texts)

        assert len(results) == 3
        assert all(len(emb) == 3 for emb in results)
        assert service.get_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list(self, mock_config: Config):
        """Test batch generation with empty list."""
        service = EmbeddingService(mock_config)

        results = await service.get_embeddings_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_get_embedding_empty_text(self, mock_config: Config):
        """Test embedding generation with empty text."""
        service = EmbeddingService(mock_config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_response = {"embedding": [0.0] * 768}
            mock_response_obj = MagicMock()
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("")

            assert isinstance(result, list)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_error_handling_network(self, mock_config: Config):
        """Test network error handling."""
        service = EmbeddingService(mock_config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.post = MagicMock(side_effect=ConnectionError("Network unreachable"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test")

            # Should return zero vector
            assert isinstance(result, list)
            assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_error_handling_invalid_response(self, mock_config: Config):
        """Test invalid response handling."""
        service = EmbeddingService(mock_config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_response_obj = MagicMock()
            # Return invalid structure
            mock_response_obj.json = AsyncMock(return_value={"error": "Invalid model"})
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test")

            # Should return zero vector
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_custom_model(self):
        """Test using custom embedding model."""
        config = Config(
            ollama_base_url="http://localhost:11434",
            embedding_model="custom-model",
        )
        service = EmbeddingService(config)

        assert service.model == "custom-model"

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_config: Config):
        """Test concurrent embedding generation."""
        service = EmbeddingService(mock_config)

        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        results = await service.get_embeddings_batch(texts)

        assert len(results) == 5
        # All should use the same mock
        assert service.get_embedding.call_count == 5
