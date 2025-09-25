"""
Simplified comprehensive test suite for buddhi_pragati.models module.
Tests core model interface functionality without real API calls or model loading.
"""

import pytest
from unittest.mock import Mock, patch

# Import modules to test
from buddhi_pragati.models.model_interface import (
    BaseModelInterface,
    UnifiedModelInterface,
    retry_with_exponential_backoff,
    validate_response
)


class TestUtilityFunctions:
    """Test utility functions used across the models module."""

    def test_validate_response_valid(self):
        """Test response validation with valid responses."""
        # Valid responses
        assert validate_response("This is a valid response with sufficient length")
        assert validate_response("Another good response that meets criteria", min_length=10)
        assert validate_response("Short but diverse text!", min_length=5)

        # Test longer response
        long_response = "This is a comprehensive response that contains multiple sentences and diverse vocabulary."
        assert validate_response(long_response, min_length=20)

    def test_validate_response_invalid(self):
        """Test response validation with invalid responses."""
        # Invalid responses
        assert not validate_response("")  # Empty
        assert not validate_response(None)  # None
        assert not validate_response("short", min_length=10)  # Too short
        assert not validate_response("aaaaaaaaaa")  # Not diverse enough
        assert not validate_response("   ")  # Just whitespace
        assert not validate_response(123)  # Not string
        assert not validate_response("   \n\n  ")  # Only whitespace and newlines

    def test_validate_response_edge_cases(self):
        """Test edge cases for response validation."""
        # Test with minimum length requirements
        assert validate_response("Hello World!", min_length=12)
        assert not validate_response("Hello World!", min_length=13)

        # Test with special characters
        assert validate_response("Response with numbers: 123 and symbols: !@#", min_length=10)

        # Test Unicode content
        assert validate_response("हैलो वर्ल्ड! This is multilingual.", min_length=15)

    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3)
        def mock_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = mock_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_failure_then_success(self):
        """Test retry decorator with failure then success."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.01)
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success after retry"

        result = mock_function()
        assert result == "success after retry"
        assert call_count == 2

    def test_retry_decorator_all_failures(self):
        """Test retry decorator with all attempts failing."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=2, initial_delay=0.01)
        def mock_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            mock_function()
        assert call_count == 2

    def test_retry_decorator_empty_response_handling(self):
        """Test retry decorator with empty responses."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.01)
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return ""  # Empty response
            return "finally got response"

        result = mock_function()
        assert result == "finally got response"
        assert call_count == 3

    def test_retry_decorator_whitespace_response(self):
        """Test retry decorator treats whitespace-only responses as failures."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.01)
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return "   \n  "  # Whitespace only
            return "real content"

        result = mock_function()
        assert result == "real content"
        assert call_count == 3

    def test_retry_decorator_custom_exceptions(self):
        """Test retry decorator with custom exception types."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_attempts=2,
            initial_delay=0.01,
            exceptions=(ValueError, TypeError)
        )
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First failure")
            return "success"

        result = mock_function()
        assert result == "success"
        assert call_count == 2

    def test_retry_decorator_unhandled_exception(self):
        """Test retry decorator doesn't retry unhandled exceptions."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_attempts=3,
            initial_delay=0.01,
            exceptions=(ValueError,)  # Only catch ValueError
        )
        def mock_function():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not handled")  # Should not be retried

        with pytest.raises(TypeError, match="Not handled"):
            mock_function()
        assert call_count == 1  # Should not retry


class TestBaseModelInterface:
    """Test BaseModelInterface abstract base class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock()
        config.get.side_effect = lambda key, default="0.1": {
            "DEFAULT_TEMPERATURE_REASONING": "0.0",
            "DEFAULT_TEMPERATURE_STANDARD": "0.1",
            "LOW_REASONING_TOKENS": "500",
            "NORMAL_REASONING_TOKENS": "1000",
            "HIGH_REASONING_TOKENS": "2000"
        }.get(key, default)
        return config

    def test_base_model_initialization(self, mock_config):
        """Test BaseModelInterface initialization."""
        # Create a concrete implementation for testing
        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            # Test initialization with defaults
            model = TestModel("test-model")
            assert model.model_name == "test-model"
            assert not model.reasoning_mode
            assert model.reasoning_effort == "normal"
            assert model.temperature == 0.1  # Standard temperature
            assert hasattr(model, 'last_token_usage')
            assert isinstance(model.last_token_usage, dict)

            # Test initialization with reasoning mode
            reasoning_model = TestModel(
                "test-model",
                reasoning_mode=True,
                reasoning_effort="high",
                temperature=0.2  # Custom temperature
            )
            assert reasoning_model.reasoning_mode
            assert reasoning_model.reasoning_effort == "high"
            assert reasoning_model.temperature == 0.2  # Custom overrides default

    def test_reasoning_token_limit_calculation(self, mock_config):
        """Test reasoning token limit calculation based on effort."""
        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            model = TestModel("test-model")

            # Test different effort levels
            assert model.get_reasoning_token_limit("low") == 500
            assert model.get_reasoning_token_limit("normal") == 1000
            assert model.get_reasoning_token_limit("high") == 2000
            assert model.get_reasoning_token_limit() == 1000  # Default normal

    def test_reasoning_token_limit_fallback(self, mock_config):
        """Test fallback when reasoning token config is missing."""
        # Mock config without reasoning token settings
        limited_config = Mock()
        limited_config.get.side_effect = lambda key, default="1000": {
            "DEFAULT_TEMPERATURE_STANDARD": "0.1"
        }.get(key, default)

        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = limited_config

            model = TestModel("test-model")

            # Should use default fallback value
            assert model.get_reasoning_token_limit("missing") == 1000

    def test_abstract_method_enforcement(self):
        """Test that BaseModelInterface cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModelInterface("test-model")

    def test_temperature_defaults(self, mock_config):
        """Test temperature defaults based on reasoning mode."""
        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            # Test standard mode default temperature
            standard_model = TestModel("test", reasoning_mode=False)
            assert standard_model.temperature == 0.1

            # Test reasoning mode default temperature
            reasoning_model = TestModel("test", reasoning_mode=True)
            assert reasoning_model.temperature == 0.0

    def test_logger_initialization(self, mock_config):
        """Test that logger is properly initialized."""
        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            model = TestModel("test-model")
            assert hasattr(model, 'logger')
            assert model.logger.name == "TestModel"


class TestUnifiedModelInterface:
    """Test UnifiedModelInterface routing and orchestration."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for UnifiedModelInterface testing."""
        config = Mock()
        config.get.return_value = "fake_key"
        return config

    def test_model_source_auto_detection(self, mock_config):
        """Test automatic model source detection."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            # Mock all backend classes to avoid initialization issues
            with patch('buddhi_pragati.models.model_interface.OpenAIBackend'), \
                 patch('buddhi_pragati.models.model_interface.AnthropicBackend'), \
                 patch('buddhi_pragati.models.model_interface.HuggingFaceBackend'), \
                 patch('buddhi_pragati.models.model_interface.SarvamAIBackend'), \
                 patch('buddhi_pragati.models.model_interface.OpenRouterBackend'):

                # Test OpenAI model detection
                openai_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o1-preview"]
                for model in openai_models:
                    interface = UnifiedModelInterface(model)
                    assert interface.source == "openai"
                    assert interface.model_name == model

                # Test Anthropic model detection
                anthropic_models = ["claude-sonnet-4", "claude-3-5-sonnet", "claude-3-haiku"]
                for model in anthropic_models:
                    interface = UnifiedModelInterface(model)
                    assert interface.source == "anthropic"

                # Test SarvamAI model detection
                sarvam_models = ["sarvam-m", "sarvam-2b"]
                for model in sarvam_models:
                    interface = UnifiedModelInterface(model)
                    assert interface.source == "sarvamai"

                # Test HuggingFace model detection (default fallback)
                hf_models = ["microsoft/DialoGPT-large", "ai4bharat/airavata"]
                for model in hf_models:
                    interface = UnifiedModelInterface(model)
                    assert interface.source == "huggingface"

                # Test OpenRouter model detection (specific prefixes)
                openrouter_models = ["meta-llama/Llama-2-7b-hf", "mistralai/mistral-large"]
                for model in openrouter_models:
                    interface = UnifiedModelInterface(model)
                    assert interface.source == "openrouter"

    def test_explicit_source_override(self, mock_config):
        """Test explicit model source specification."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with patch('buddhi_pragati.models.model_interface.OpenRouterBackend'):
                # Test explicit source override
                interface = UnifiedModelInterface("some-model", source="openrouter")
                assert interface.source == "openrouter"
                assert interface.model_name == "some-model"

    def test_unsupported_source_handling(self, mock_config):
        """Test handling of unsupported model sources."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="Unsupported source"):
                UnifiedModelInterface("some-model", source="unsupported")

    def test_backend_initialization_parameters(self, mock_config):
        """Test that parameters are properly passed to backends."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with patch('buddhi_pragati.models.model_interface.OpenAIBackend') as mock_openai:
                # Mock backend instance
                mock_backend = Mock()
                mock_openai.return_value = mock_backend

                interface = UnifiedModelInterface(
                    "gpt-4o",
                    reasoning_mode=True,
                    reasoning_effort="high",
                    temperature=0.3
                )

                # Verify backend was called with correct parameters
                mock_openai.assert_called_once_with(
                    "gpt-4o",
                    reasoning_mode=True,
                    reasoning_effort="high",
                    temperature=0.3
                )

                assert interface.backend == mock_backend

    def test_generate_response_delegation(self, mock_config):
        """Test that generate_response is properly delegated to backend."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with patch('buddhi_pragati.models.model_interface.OpenAIBackend') as mock_backend_class:
                # Mock backend instance
                mock_backend = Mock()
                mock_backend.generate_response.return_value = "test response from backend"
                mock_backend_class.return_value = mock_backend

                interface = UnifiedModelInterface("gpt-4o")
                response = interface.generate_response(
                    "test prompt",
                    max_tokens=500,
                    reasoning_mode=True
                )

                # Verify backend method was called with correct parameters
                mock_backend.generate_response.assert_called_once_with(
                    prompt="test prompt",
                    max_completion_tokens=1000,  # Default value
                    max_tokens=500,
                    reasoning_mode=True,
                    reasoning_effort=None
                )
                assert response == "test response from backend"

    def test_model_source_detection_edge_cases(self, mock_config):
        """Test edge cases in model source detection."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with patch('buddhi_pragati.models.model_interface.OpenAIBackend'), \
                 patch('buddhi_pragati.models.model_interface.OpenRouterBackend'):

                # Test models that might be ambiguous
                interface = UnifiedModelInterface("gpt-4-turbo")
                assert interface.source == "openai"

                # Test explicit openrouter override for openai-like model
                interface = UnifiedModelInterface("gpt-4", source="openrouter")
                assert interface.source == "openrouter"

    def test_initialization_with_kwargs(self, mock_config):
        """Test initialization with additional keyword arguments."""
        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            with patch('buddhi_pragati.models.model_interface.OpenAIBackend') as mock_backend:
                _ = UnifiedModelInterface(
                    "gpt-4o",
                    custom_param="custom_value",
                    another_param=123
                )

                # Verify backend was called with kwargs
                call_args = mock_backend.call_args
                assert "custom_param" in call_args[1]
                assert call_args[1]["custom_param"] == "custom_value"
                assert call_args[1]["another_param"] == 123


class TestModelInterfaceIntegration:
    """Test integration scenarios and cross-cutting concerns."""

    def test_configuration_loading_consistency(self):
        """Test that configuration is loaded consistently across components."""
        mock_config = Mock()
        config_values = {
            'DEFAULT_TEMPERATURE_STANDARD': '0.1',
            'DEFAULT_TEMPERATURE_REASONING': '0.0',
            'NORMAL_REASONING_TOKENS': '1000',
            'DEFAULT_OPENAI_API_KEY': 'openai_key'
        }
        mock_config.get.side_effect = lambda key, default=None: config_values.get(key, default)

        # Create test model
        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            model = TestModel("test-model")

            # Verify config is accessible
            assert model.config == mock_config
            assert model.temperature == 0.1  # From config
            assert model.get_reasoning_token_limit() == 1000  # From config

    def test_parameter_override_precedence(self):
        """Test that explicit parameters override config defaults."""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            'DEFAULT_TEMPERATURE_STANDARD': '0.1',
            'DEFAULT_TEMPERATURE_REASONING': '0.0'
        }.get(key, default)

        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config

            # Test explicit parameter overrides config
            model = TestModel("test-model", temperature=0.5)
            assert model.temperature == 0.5  # Explicit value, not config default

    def test_error_handling_consistency(self):
        """Test consistent error handling patterns across the interface."""
        # Test that missing config keys don't break initialization with better fallbacks
        empty_config = Mock()
        empty_config.get.side_effect = lambda key, default=None: {
            'DEFAULT_TEMPERATURE_STANDARD': '0.1',  # Provide essential defaults
            'DEFAULT_TEMPERATURE_REASONING': '0.0'
        }.get(key, default)

        class TestModel(BaseModelInterface):
            def generate_response(self, prompt, **kwargs):
                return "test response"

        with patch('buddhi_pragati.models.model_interface.get_config') as mock_get_config:
            mock_get_config.return_value = empty_config

            # Should still initialize with fallback defaults
            model = TestModel("test-model")
            assert model.model_name == "test-model"
            assert model.temperature is not None  # Should have some default
            assert model.temperature == 0.1  # Should use standard temperature


if __name__ == "__main__":
    pytest.main([__file__])
