"""
Tests for Embedding Service (services/embedding.py).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_reminiscence.config.settings import Config
from agent_reminiscence.services.embedding import EmbeddingService


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


class TestMultiProviderEmbedding:
    """Test multi-provider embedding support."""

    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization."""
        config = Config(
            embedding_provider="openai",
            openai_api_key="test-key",
            openai_embedding_model="text-embedding-3-small",
        )
        service = EmbeddingService(config)

        assert service.provider.value == "openai"
        assert service.model == "text-embedding-3-small"

    def test_google_provider_initialization(self):
        """Test Google provider initialization."""
        config = Config(
            embedding_provider="google",
            google_api_key="test-key",
            google_embedding_model="text-embedding-004",
        )
        service = EmbeddingService(config)

        assert service.provider.value == "google"
        assert service.model == "text-embedding-004"

    def test_ollama_provider_default(self):
        """Test default Ollama provider."""
        config = Config(embedding_provider="ollama")
        service = EmbeddingService(config)

        assert service.provider.value == "ollama"

    def test_invalid_provider_fallback(self):
        """Test invalid provider falls back to Ollama."""
        config = Config(embedding_provider="invalid_provider")
        service = EmbeddingService(config)

        assert service.provider.value == "ollama"

    @pytest.mark.asyncio
    async def test_openai_embedding_routing(self):
        """Test embeddings route through OpenAI when configured."""
        config = Config(
            embedding_provider="openai",
            openai_api_key="test-key",
            vector_dimension=1536,
        )
        service = EmbeddingService(config)

        mock_response = {"data": [{"embedding": [0.1] * 1536}]}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)
            mock_response_obj.raise_for_status = MagicMock()

            mock_session.post = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test text")

            assert len(result) == 1536
            # Verify OpenAI endpoint was called
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert "api.openai.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_google_embedding_routing(self):
        """Test embeddings route through Google when configured."""
        config = Config(
            embedding_provider="google",
            google_api_key="test-key",
            google_embedding_model="text-embedding-004",
            vector_dimension=768,
        )
        service = EmbeddingService(config)

        mock_response = {"embedding": {"values": [0.1] * 768}}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)
            mock_response_obj.raise_for_status = MagicMock()

            mock_session.post = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await service.get_embedding("test text")

            assert len(result) == 768
            # Verify Google endpoint was called
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert "generativelanguage.googleapis.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_openai_missing_api_key(self):
        """Test OpenAI returns zero vector when API key missing."""
        config = Config(embedding_provider="openai", openai_api_key=None)
        service = EmbeddingService(config)

        result = await service.get_embedding("test text")

        assert result == [0.0] * config.vector_dimension

    @pytest.mark.asyncio
    async def test_google_missing_api_key(self):
        """Test Google returns zero vector when API key missing."""
        config = Config(embedding_provider="google", google_api_key=None)
        service = EmbeddingService(config)

        result = await service.get_embedding("test text")

        assert result == [0.0] * config.vector_dimension


