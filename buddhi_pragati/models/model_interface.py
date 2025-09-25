"""
Unified model interface system supporting multiple providers with modern API clients.

This module provides a comprehensive model interface that supports:
- OpenAI (Responses API for gpt-4.1/gpt-5/o1/o3, Chat Completions for others)
- Anthropic (Messages API with thinking mode support and official client option)
- OpenRouter (proxy for multiple models with enhanced error handling)
- HuggingFace (local and hosted models with automatic architecture detection)
- SarvamAI (official client library with api_subscription_key authentication)

Key Features:
- Modern API client integration with fallback to requests for backward compatibility
- Reasoning mode support across all providers with proper parameter mapping
- Enhanced error handling and response validation
- Proper token usage tracking for cost analysis
- Configuration-driven parameter management from crossword_config.txt
- Retry logic with exponential backoff for robustness

API Implementation Details:
- OpenAI: Supports both new Responses API and legacy Chat Completions
- Anthropic: Fixed thinking mode parameter format and response content parsing
- SarvamAI: Complete rewrite using official client with proper api_subscription_key handling
- All backends maintain backward compatibility while supporting modern features
"""

import logging
import requests
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from functools import wraps

from ..utils.config_loader import get_config

# Optional imports for specific providers
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    pipeline = None


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    # Check if response is truly empty (no content at all)
                    if result is not None and result.strip():
                        # Return any non-empty response - let validation handle quality
                        return result
                    # Truly empty response (None or only whitespace) - retry
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Truly empty response from {func.__name__}, retrying... (attempt {attempt + 1}/{max_attempts})"
                        )
                    else:
                        logging.error(
                            f"Truly empty response from {func.__name__} after {max_attempts} attempts"
                        )
                        return ""

                except exceptions as e:
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logging.error(
                            f"All {max_attempts} attempts failed. Last error: {e}"
                        )
                        raise

            return ""

        return wrapper

    return decorator


def validate_response(response: str, min_length: int = 10) -> bool:
    """
    Validate that a model response is meaningful.

    Args:
        response: The response text to validate
        min_length: Minimum acceptable response length

    Returns:
        True if response is valid, False otherwise
    """
    if not response or not isinstance(response, str):
        return False

    cleaned = response.strip()
    if len(cleaned) < min_length:
        return False

    # Check for repetitive patterns (signs of failure)
    if len(set(cleaned)) < 5:  # Too few unique characters
        return False

    return True


class BaseModelInterface(ABC):
    """
    Base interface that all model types must implement.

    Supports reasoning modes, proper configuration reading, and
    experimental parameters for comprehensive evaluation.
    """

    def __init__(
        self,
        model_name: str,
        reasoning_mode: bool = False,
        reasoning_effort: str = "normal",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.reasoning_mode = reasoning_mode
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.config = get_config()

        # Token usage tracking
        self.last_token_usage = {}

        # Set default temperature based on reasoning mode if not provided
        if self.temperature is None:
            if self.reasoning_mode:
                self.temperature = float(
                    self.config.get("DEFAULT_TEMPERATURE_REASONING", "0.0")
                )
            else:
                self.temperature = float(
                    self.config.get("DEFAULT_TEMPERATURE_STANDARD", "0.1")
                )

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate response to prompt with optional reasoning parameters.

        Args:
            prompt: Input prompt text
            max_completion_tokens: Max tokens for completion (used by some models)
            max_tokens: Max total tokens (used by most models)
            reasoning_mode: Override instance reasoning mode
            reasoning_effort: Override instance reasoning effort

        Returns:
            Generated response text
        """
        pass

    def get_reasoning_token_limit(self, effort: Optional[str] = None) -> int:
        """Get token limit based on reasoning effort level."""
        effort = effort or self.reasoning_effort
        effort_key = f"{effort.upper()}_REASONING_TOKENS"
        return int(self.config.get(effort_key, "1000"))

    def get_last_token_usage(self) -> Dict[str, Any]:
        """Return token usage from last API call."""
        return self.last_token_usage.copy()


class UnifiedModelInterface:
    """
    Unified interface that can handle multiple model sources with reasoning support.

    Usage examples:
    # OpenAI with reasoning mode
    model = UnifiedModelInterface("o1", source="openai", reasoning_mode=True)

    # Anthropic with thinking mode
    model = UnifiedModelInterface("claude-sonnet-4", source="anthropic", reasoning_mode=True)

    # SarvamAI for Indic languages
    model = UnifiedModelInterface("sarvam-m", source="sarvamai", reasoning_mode=True)

    # OpenRouter proxy
    model = UnifiedModelInterface("mistralai/mistral-small", source="openrouter")

    # HuggingFace local model
    model = UnifiedModelInterface("ai4bharat/Airavata", source="huggingface", device="cpu")
    """

    def __init__(
        self,
        model_name: str,
        source: Optional[str] = None,
        reasoning_mode: bool = False,
        reasoning_effort: str = "normal",
        **kwargs,
    ):
        """
        Initialize unified model interface.

        Args:
            model_name: Name of the model
            source: Provider source ("openai", "anthropic", "openrouter", "huggingface", "sarvamai")
            reasoning_mode: Enable reasoning mode for supported models
            reasoning_effort: Reasoning effort level ("low", "normal", "high")
            **kwargs: Additional arguments (api_key, device, temperature, etc.)
        """
        self.model_name = model_name
        self.reasoning_mode = reasoning_mode
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger("UnifiedModelInterface")
        self.config = get_config()

        # Auto-detect source if not provided
        if source is None:
            source = self._detect_model_source(model_name)

        self.source = source.lower()

        # Initialize the appropriate backend
        if self.source == "openai":
            self.backend = OpenAIBackend(
                model_name,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
        elif self.source == "anthropic":
            self.backend = AnthropicBackend(
                model_name,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
        elif self.source == "sarvamai":
            self.backend = SarvamAIBackend(
                model_name,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
        elif self.source == "openrouter":
            self.backend = OpenRouterBackend(
                model_name,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
        elif self.source == "huggingface":
            try:
                self.backend = HuggingFaceBackend(
                    model_name,
                    reasoning_mode=reasoning_mode,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                )
            except ImportError as e:
                # Handle missing transformers/torch gracefully
                self.logger.error(f"HuggingFace model {model_name} unavailable: {e}")
                raise ImportError("HuggingFace transformers library not available. Install with: pip install transformers torch") from e
        else:
            raise ValueError(
                f"Unsupported source: {source}. Use 'openai', 'anthropic', 'sarvamai', 'openrouter', or 'huggingface'"
            )

        self.logger.info(
            f"Initialized {self.source} model: {model_name} (reasoning: {reasoning_mode})"
        )

    def _detect_model_source(self, model_name: str) -> str:
        """Auto-detect model source from model name patterns."""
        # OpenAI models
        if model_name.startswith(("gpt-", "o1", "o3")):
            return "openai"
        # Anthropic models
        elif model_name.startswith("claude"):
            return "anthropic"
        # SarvamAI models
        elif "sarvam" in model_name.lower():
            return "sarvamai"
        # HuggingFace models (based on models_description.txt)
        elif model_name.startswith(
            (
                "ai4bharat/",
                "CohereForAI/",
                "Cognitive-Lab/",
                "nickmalhotra/",
                "smallstepai/",
                "abhinand/",
                "openai/gpt-oss",
            )
        ) or (model_name.startswith("sarvamai/") and "OpenHathi" in model_name):
            return "huggingface"
        # OpenRouter models (based on models_description.txt)
        elif model_name.startswith(
            (
                "deepseek/",
                "google/gemini",
                "meta-llama/",
                "mistralai/mistral",
                "moonshotai/",
                "qwen/",
            )
        ):
            return "openrouter"
        # Default to HuggingFace for models without clear indicators
        else:
            return "huggingface"

    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using the configured backend."""
        return self.backend.generate_response(
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
        )

    def get_last_token_usage(self) -> Dict[str, Any]:
        """Get token usage from last API call."""
        return self.backend.get_last_token_usage()

    def cleanup_model(self):
        """Clean up model and free memory resources."""
        if hasattr(self.backend, 'cleanup_model'):
            self.backend.cleanup_model()
        else:
            self.logger.info(f"Cleanup not supported for {self.source} backend")


class HuggingFaceBackend(BaseModelInterface):
    """HuggingFace model backend with automatic architecture detection and pipeline support."""

    def __init__(self, model_name: str, device: str = "auto", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers library not available. "
                "Install with: pip install transformers torch"
            )

        # Get HuggingFace token with proper fallback hierarchy: explicit > config > env
        self.hf_token = (
            api_key
            or self.config.get("DEFAULT_HF_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

        # Set token for HuggingFace hub if available
        if self.hf_token:
            try:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
                self.logger.info("Successfully authenticated with HuggingFace Hub")
            except Exception as e:
                self.logger.warning(f"Failed to authenticate with HuggingFace Hub: {e}")
        else:
            self.logger.warning("No HuggingFace token found - some models may not be accessible")

        self.device = self._determine_device(device)
        self.pipeline = None
        self.model_type = None
        self.max_input_length = 1024  # Default, will be updated in _load_tokenizer
        self._load_model()

    def _determine_device(self, requested_device: str) -> str:
        """Determine the best available device for model execution."""
        if requested_device != "auto":
            return requested_device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _detect_model_architecture(self) -> str:
        """Detect the specific model architecture for specialized loading."""
        model_name_lower = self.model_name.lower()

        # Model-specific architecture detection based on research findings
        if "aya-101" in model_name_lower:
            return "seq2seq"  # T5-based sequence-to-sequence
        elif any(x in model_name_lower for x in ["llama", "ambari", "tamil"]):
            return "llama"    # LLaMA-based models
        else:
            return "auto"     # Use automatic detection

    def _detect_model_type(self) -> str:
        """Detect the model architecture type for pipeline creation."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            model_type = config.model_type.lower()

            # Enhanced model type detection based on architecture research
            architecture = self._detect_model_architecture()

            if architecture == "seq2seq":
                return "text2text-generation"  # Aya-101 and similar
            elif model_type in ["t5", "mt5", "bart", "mbart", "pegasus", "marian"]:
                return "text2text-generation"
            elif model_type in ["bert", "roberta", "electra", "deberta"]:
                return "fill-mask"  # BERT-style models
            else:
                return "text-generation"  # Default to causal LM

        except Exception as e:
            self.logger.warning(
                f"Could not detect model type, defaulting to text-generation: {e}"
            )
            return "text-generation"

    def _load_model(self):
        """Load HuggingFace model using pipeline with enhanced architecture handling."""
        try:
            import warnings

            warnings.filterwarnings("ignore", category=UserWarning)

            self.logger.info(f"Loading HuggingFace model: {self.model_name}")

            # Detect model architecture and type
            self.architecture = self._detect_model_architecture()
            self.model_type = self._detect_model_type()
            self.logger.info(f"Detected architecture: {self.architecture}, type: {self.model_type}")

            # Enhanced model loading based on architecture
            model_kwargs = self._prepare_model_kwargs()
            tokenizer = self._load_tokenizer()

            # Create pipeline with architecture-specific handling
            try:
                self.pipeline = self._create_pipeline(tokenizer, model_kwargs)
                self.logger.info(f"Pipeline created successfully for {self.model_type}")

            except Exception as e:
                # Fallback: Try simpler configuration
                self.logger.warning(f"Failed to load with optimal config, trying fallback: {e}")
                self.pipeline = self._create_fallback_pipeline(tokenizer)
                self.device = "cpu"

            self.logger.info(f"HuggingFace model loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace models. Install with: pip install transformers torch"
            )
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            # Provide helpful error message
            if "MXFP4" in str(e) or "quantized" in str(e).lower():
                raise RuntimeError(
                    f"Model {self.model_name} requires GPU for quantized inference. "
                    "Please use a different model or run on a GPU-enabled system."
                )
            raise

    def _prepare_model_kwargs(self) -> dict:
        """Prepare model loading arguments based on architecture and device."""
        model_kwargs = {}

        try:
            import torch
        except ImportError:
            # Fallback to CPU with minimal configuration
            return {"device_map": "cpu"}

        # Memory-efficient loading strategy
        if self.device == "cpu":
            # CPU-only configuration - avoid device conflicts
            model_kwargs["device_map"] = "cpu"
            model_kwargs["dtype"] = torch.float32
            # Low memory mode for CPU
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            # GPU/MPS configuration with memory optimization
            # Use device_map for automatic device placement (don't set device parameter)
            is_large_model = (self.architecture == "llama" or
                            any(size in self.model_name.lower() for size in ["7b", "8b", "13b", "70b"]))

            if is_large_model:
                # Large models: use auto device mapping for memory efficiency
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = {0: "80%", "cpu": "50%"}  # Reserve memory
            elif self.device == "cuda" and torch.cuda.is_available():
                # Small models: can fit on single GPU
                model_kwargs["device_map"] = {"":  0}  # Single GPU placement
            elif self.device == "mps":
                # MPS device mapping
                model_kwargs["device_map"] = {"":  "mps"}
            else:
                # Fallback to CPU
                model_kwargs["device_map"] = "cpu"
                model_kwargs["low_cpu_mem_usage"] = True

            # Enhanced precision handling based on model research
            if any(x in self.model_name.lower() for x in ["misal", "tamil"]):
                # Models that prefer bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    model_kwargs["dtype"] = torch.bfloat16
                else:
                    model_kwargs["dtype"] = torch.float16
            elif self.device in ["cuda", "mps"] and torch.cuda.is_available():
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float32

        # Additional memory optimizations
        model_kwargs["trust_remote_code"] = True

        return model_kwargs

    def _load_tokenizer(self):
        """Load tokenizer with architecture-specific handling and sequence length management."""
        try:
            # Architecture-specific tokenizer loading
            if self.architecture == "llama":
                try:
                    from transformers import LlamaTokenizer
                    tokenizer = LlamaTokenizer.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                    self.logger.info(f"Loaded LlamaTokenizer for {self.model_name}")
                except Exception:
                    # Fallback to AutoTokenizer
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )

            # Enhanced tokenizer configuration
            if tokenizer.pad_token is None:
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = "[PAD]"

            # Special handling for chat/instruct models
            if any(x in self.model_name.lower() for x in ["instruct", "chat", "airavata", "projectindus"]):
                tokenizer.padding_side = "left"  # Better for instruct models

            # Store model max length for sequence handling
            self.max_input_length = getattr(tokenizer, 'model_max_length', 1024)
            if self.max_input_length > 100000:  # Some tokenizers have very large defaults
                # Set reasonable limits based on model architecture
                if "aya-101" in self.model_name.lower():
                    self.max_input_length = 1024  # T5-based models
                elif "airavata" in self.model_name.lower():
                    self.max_input_length = 2048  # LLaMA-based models
                else:
                    self.max_input_length = 1024  # Conservative default

            self.logger.info(f"Set maximum input length: {self.max_input_length}")
            return tokenizer

        except Exception as e:
            self.logger.warning(f"Tokenizer loading issue: {e}")
            return None

    def _create_pipeline(self, tokenizer, model_kwargs):
        """Create pipeline with architecture-specific configuration."""
        # Filter model_kwargs to only include pipeline-compatible parameters
        # Remove model loading parameters that pipeline doesn't accept
        pipeline_compatible_kwargs = {}
        pipeline_forbidden_params = {
            "low_cpu_mem_usage", "trust_remote_code", "return_full_text",
            "max_memory", "offload_folder", "offload_state_dict", "device_map"
        }

        for key, value in model_kwargs.items():
            if key not in pipeline_forbidden_params:
                pipeline_compatible_kwargs[key] = value

        # Pipeline parameters - avoid device conflicts with device_map
        pipeline_kwargs = {
            "task": self.model_type,
            "model": self.model_name,
            "tokenizer": tokenizer,
            "trust_remote_code": True,
            **pipeline_compatible_kwargs,
        }

        # IMPORTANT: Don't set device parameter when using device_map
        # This avoids the "Both device and device_map are specified" error
        if "device_map" in model_kwargs:
            # Remove any device parameter to avoid conflicts
            pipeline_kwargs.pop("device", None)
        else:
            # Only set device if device_map is not used
            pipeline_kwargs["device"] = self.device if self.device != "cpu" else -1

        if self.architecture == "llama":
            # Explicit LLaMA model loading for better compatibility
            try:
                from transformers import LlamaForCausalLM
                pipeline_kwargs["model_class"] = LlamaForCausalLM
                return pipeline(**pipeline_kwargs)
            except Exception as e:
                self.logger.warning(f"LLaMA-specific loading failed: {e}, using standard pipeline")
                # Remove model_class and fallback to standard pipeline
                pipeline_kwargs.pop("model_class", None)

        # Standard pipeline creation
        return pipeline(**pipeline_kwargs)

    def _create_fallback_pipeline(self, tokenizer):
        """Create fallback pipeline with minimal configuration."""
        try:
            import torch
        except ImportError:
            torch = None

        fallback_kwargs = {
            "task": self.model_type,
            "model": self.model_name,
            "tokenizer": tokenizer,
            "trust_remote_code": True,
            "device": -1,  # CPU device for fallback
        }

        if torch:
            fallback_kwargs["dtype"] = torch.float32

        return pipeline(**fallback_kwargs)

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template based on model-specific requirements."""
        model_name_lower = self.model_name.lower()

        # Model-specific chat template formatting based on research
        if "airavata" in model_name_lower:
            # Airavata uses <|user|>/<|assistant|> format
            return f"<|user|>\n{prompt}<|assistant|>\n"
        elif "projectindus" in model_name_lower:
            # ProjectIndus - Try to use tokenizer's chat template if available
            if hasattr(self, 'pipeline') and self.pipeline and hasattr(self.pipeline.tokenizer, 'apply_chat_template'):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    return self.pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to apply chat template for ProjectIndus: {e}")
            # Fallback to simple conversation format
            return f"Human: {prompt}\nAssistant:"
        elif "misal" in model_name_lower:
            # Misal uses instruct format
            return f"### Instruction:\n{prompt}\n\n### Response:"
        elif "tamil" in model_name_lower and "llama" in model_name_lower:
            # Tamil LLaMA uses ChatML format
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "ambari" in model_name_lower:
            # Ambari uses standard instruct format
            return f"[INST] {prompt} [/INST]"
        else:
            # Default: return prompt as-is
            return prompt

    def _get_model_specific_params(self, use_reasoning: bool, effort: str, max_tokens: int) -> dict:
        """Get model-specific generation parameters based on research findings."""
        model_name_lower = self.model_name.lower()

        # Base parameters
        temperature = 0.1 if use_reasoning else self.temperature
        max_new_tokens = (
            self.get_reasoning_token_limit(effort) if use_reasoning else max_tokens
        )

        # Ensure temperature is valid (avoid 0.0 which can cause issues)
        temperature = max(temperature, 0.01)

        # Model-specific parameter optimization with seq2seq fixes
        if "airavata" in model_name_lower:
            params = {
                "max_new_tokens": min(max_new_tokens, 250),  # Airavata optimal limit
                "do_sample": False,  # Airavata performs better without sampling
                "repetition_penalty": 1.0,
            }
            # Only add temperature if sampling is enabled
            if params["do_sample"]:
                params["temperature"] = temperature
            return params
        elif "misal" in model_name_lower:
            return {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,  # Required for temperature to work
                "repetition_penalty": 1.1,
            }
        elif "tamil" in model_name_lower and "llama" in model_name_lower:
            return {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.6,  # Tamil LLaMA optimal temperature
                "do_sample": True,  # Required for temperature to work
                "repetition_penalty": 1.1,
            }
        elif "projectindus" in model_name_lower:
            # ProjectIndus (GPT2-based) - Use parameters from model documentation
            return {
                "max_new_tokens": min(max_new_tokens, 1024),  # From docs: max_length=1024
                "temperature": 0.7,  # From docs: optimal temperature
                "do_sample": True,  # Required for temperature and other sampling params
                "top_k": 50,  # From docs: top_k=50
                "top_p": 0.95,  # From docs: top_p=0.95
                "num_beams": 1,  # Use greedy for crosswords (num_beams=5 is for generation)
                "repetition_penalty": 1.0,
            }
        elif "aya-101" in model_name_lower:
            # Aya-101 (T5-based) - Remove repetition_penalty (incompatible with T5)
            return {
                "max_new_tokens": min(max_new_tokens, 128),  # Aya-101 optimal limit
                "temperature": temperature,
                "do_sample": True,  # Required for temperature to work with T5-based models
                # Note: repetition_penalty removed - incompatible with T5 models
            }
        else:
            # Default parameters - handle seq2seq models properly
            use_sampling = temperature > 0.1 or self.model_type == "text2text-generation"
            params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": use_sampling,
                "repetition_penalty": 1.0 if use_reasoning else 1.1,
            }
            # Only add temperature if sampling is enabled
            if use_sampling:
                params["temperature"] = temperature
            return params

    def _post_process_response(self, generated_text: str, original_prompt: str) -> str:
        """Post-process generated response based on model-specific patterns."""
        if not generated_text:
            return generated_text

        model_name_lower = self.model_name.lower()

        # Remove input prompt if included
        if generated_text.startswith(original_prompt):
            generated_text = generated_text[len(original_prompt):].strip()

        # Model-specific post-processing
        if "misal" in model_name_lower:
            # Misal: Split on "### Response:" marker
            if "### Response:" in generated_text:
                generated_text = generated_text.split("### Response:")[-1].strip()
        elif "tamil" in model_name_lower and "llama" in model_name_lower:
            # Tamil LLaMA: Remove ChatML end markers
            if "<|im_end|>" in generated_text:
                generated_text = generated_text.split("<|im_end|>")[0].strip()
        elif "airavata" in model_name_lower:
            # Airavata: Clean up assistant markers
            if "<|assistant|>" in generated_text:
                generated_text = generated_text.split("<|assistant|>")[-1].strip()
        elif "aya-101" in model_name_lower:
            # Clean up any task prefixes
            for prefix in ["Answer the following:", "Translate:", "Summarize:"]:
                if generated_text.startswith(prefix):
                    generated_text = generated_text[len(prefix):].strip()

        return generated_text.strip()

    def _validate_generation_parameters(self, gen_kwargs: dict) -> dict:
        """Validate and fix generation parameters for compatibility."""
        validated_kwargs = gen_kwargs.copy()

        # Check do_sample and temperature compatibility
        if not validated_kwargs.get("do_sample", True) and "temperature" in validated_kwargs:
            self.logger.debug("Removing temperature parameter when do_sample=False")
            validated_kwargs.pop("temperature", None)
            # Also remove other sampling-related parameters
            for param in ["top_k", "top_p"]:
                if param in validated_kwargs:
                    validated_kwargs.pop(param)
                    self.logger.debug(f"Removed sampling parameter {param} when do_sample=False")

        # Check architecture-specific parameter compatibility
        if self.model_type == "text2text-generation":
            # T5-based models (like Aya-101) - ensure incompatible parameters are removed
            invalid_t5_params = ["repetition_penalty", "pad_token_id"]
            for param in invalid_t5_params:
                if param in validated_kwargs:
                    removed_value = validated_kwargs.pop(param)
                    self.logger.debug(f"Removed T5-incompatible parameter {param}={removed_value}")

            # Ensure max_length is used instead of max_new_tokens for seq2seq
            if "max_new_tokens" in validated_kwargs:
                max_tokens = validated_kwargs.pop("max_new_tokens")
                validated_kwargs["max_length"] = max_tokens
                self.logger.debug("Converted max_new_tokens to max_length for seq2seq model")

        # Validate temperature range
        if "temperature" in validated_kwargs:
            temp = validated_kwargs["temperature"]
            if temp <= 0:
                validated_kwargs["temperature"] = 0.01  # Minimum valid temperature
                self.logger.debug(f"Adjusted temperature from {temp} to 0.01")
            elif temp > 2.0:
                validated_kwargs["temperature"] = 2.0  # Maximum reasonable temperature
                self.logger.debug(f"Adjusted temperature from {temp} to 2.0")

        # Validate sampling parameters
        if "top_p" in validated_kwargs:
            top_p = validated_kwargs["top_p"]
            if top_p <= 0 or top_p > 1:
                validated_kwargs["top_p"] = 0.95  # Default safe value
                self.logger.debug(f"Adjusted top_p from {top_p} to 0.95")

        if "top_k" in validated_kwargs:
            top_k = validated_kwargs["top_k"]
            if top_k <= 0:
                validated_kwargs.pop("top_k")  # Remove invalid top_k
                self.logger.debug(f"Removed invalid top_k={top_k}")

        return validated_kwargs

    def _handle_sequence_length(self, formatted_prompt: str) -> str:
        """Handle sequence length limits with intelligent truncation."""
        if not hasattr(self, 'pipeline') or not self.pipeline or not self.pipeline.tokenizer:
            return formatted_prompt

        try:
            tokenizer = self.pipeline.tokenizer

            # Tokenize to check length
            tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)

            # Check if truncation is needed
            if len(tokens) <= self.max_input_length:
                return formatted_prompt

            # Calculate target length (leave room for generation)
            target_length = max(self.max_input_length - 100, self.max_input_length // 2)

            self.logger.warning(
                f"Input length {len(tokens)} exceeds maximum {self.max_input_length}, "
                f"truncating to {target_length} tokens"
            )

            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:target_length]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

            # For chat templates, try to preserve the format
            if any(marker in formatted_prompt for marker in ["<|user|>", "### Instruction:", "[INST]"]):
                # Try to find the actual content part and truncate that
                lines = truncated_text.split('\n')
                if len(lines) > 1:
                    # Keep the template markers and truncate the content
                    template_part = lines[0]
                    content_parts = lines[1:]
                    truncated_content = '\n'.join(content_parts)[:len(truncated_text) - len(template_part) - 1]
                    truncated_text = f"{template_part}\n{truncated_content}"

            return truncated_text

        except Exception as e:
            self.logger.error(f"Error handling sequence length: {e}")
            # Fallback: simple character-based truncation
            if len(formatted_prompt) > self.max_input_length * 4:  # Rough estimate of 4 chars per token
                return formatted_prompt[:self.max_input_length * 4]
            return formatted_prompt

    @retry_with_exponential_backoff(max_attempts=3, exceptions=(Exception,))
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        max_completion_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using HuggingFace pipeline with enhanced model-specific handling."""
        try:
            if self.pipeline is None:
                self.logger.error("Pipeline not initialized")
                return ""

            start_time = time.time()
            self.logger.info(
                f"Generating with {self.model_type} pipeline for {self.model_name} (arch: {self.architecture})"
            )

            # Use reasoning mode and effort overrides if provided
            use_reasoning = (
                reasoning_mode if reasoning_mode is not None else self.reasoning_mode
            )
            effort = reasoning_effort or self.reasoning_effort

            # Apply chat template for instruct models
            formatted_prompt = self._apply_chat_template(prompt)
            self.logger.debug(f"Formatted prompt: {formatted_prompt[:100]}...")

            # Handle sequence length limits with truncation
            formatted_prompt = self._handle_sequence_length(formatted_prompt)

            # Get model-specific generation parameters
            gen_kwargs = self._get_model_specific_params(use_reasoning, effort, max_tokens)

            # Add tokenizer-specific parameters
            if hasattr(self.pipeline.tokenizer, 'pad_token_id') and self.pipeline.tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.pipeline.tokenizer.pad_token_id
            if hasattr(self.pipeline.tokenizer, 'eos_token_id') and self.pipeline.tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.pipeline.tokenizer.eos_token_id

            # Validate and fix parameter compatibility issues
            gen_kwargs = self._validate_generation_parameters(gen_kwargs)

            # Add task prefix for T5 models if needed
            if self.model_type == "text2text-generation" and "aya" in self.model_name.lower():
                formatted_prompt = f"Answer the following: {formatted_prompt}"

            # Generate response
            self.logger.debug(f"Generation parameters: {gen_kwargs}")

            try:
                # Prepare pipeline call arguments - avoid return_full_text for incompatible models
                pipeline_call_kwargs = gen_kwargs.copy()

                # Only add return_full_text for models that support it (causal LM models)
                if self.model_type == "text-generation":
                    pipeline_call_kwargs["return_full_text"] = False  # Only return generated text

                result = self.pipeline(
                    formatted_prompt,
                    **pipeline_call_kwargs,
                )

                # Extract text from result
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        generated_text = result[0].get("generated_text", "") or result[0].get("summary_text", "")
                    else:
                        generated_text = str(result[0])
                elif isinstance(result, dict):
                    generated_text = result.get("generated_text", "") or result.get("summary_text", "")
                else:
                    generated_text = str(result)

            except Exception as e:
                error_msg = str(e).lower()
                self.logger.error(f"Pipeline generation failed: {e}")

                # Handle specific error types
                if "out of memory" in error_msg or "oom" in error_msg:
                    self.logger.warning("Out of memory error, trying with reduced parameters...")
                    # Reduce max tokens and try again
                    simple_kwargs = {
                        "max_new_tokens": min(gen_kwargs.get("max_new_tokens", max_tokens), 50),
                        "do_sample": False,  # Disable sampling for memory efficiency
                    }
                    # Add tokenizer-specific parameters if available
                    if hasattr(self.pipeline.tokenizer, 'pad_token_id') and self.pipeline.tokenizer.pad_token_id is not None:
                        simple_kwargs["pad_token_id"] = self.pipeline.tokenizer.pad_token_id
                elif "invalid" in error_msg and "generation" in error_msg:
                    self.logger.warning("Invalid generation parameters, using minimal configuration...")
                    # Use absolute minimal parameters
                    simple_kwargs = {"max_new_tokens": min(max_tokens, 50)}
                else:
                    # Generic error, try simpler generation
                    simple_kwargs = {
                        "max_new_tokens": gen_kwargs.get("max_new_tokens", max_tokens),
                        "do_sample": False,
                    }

                try:
                    result = self.pipeline(formatted_prompt, **simple_kwargs)
                    # Handle different result formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            generated_text = result[0].get("generated_text", "") or result[0].get("summary_text", "")
                        else:
                            generated_text = str(result[0])
                    elif isinstance(result, dict):
                        generated_text = result.get("generated_text", "") or result.get("summary_text", "")
                    else:
                        generated_text = str(result)
                except Exception as e2:
                    self.logger.error(f"Fallback generation also failed: {e2}")
                    return ""  # Give up after fallback fails

            elapsed = time.time() - start_time
            self.logger.info(f"Generation completed in {elapsed:.2f} seconds")

            # Post-process response
            generated_text = self._post_process_response(generated_text, formatted_prompt)

            # Validate response
            if not validate_response(generated_text, min_length=5):
                self.logger.warning(
                    f"Invalid response from {self.model_name}: {generated_text[:100] if generated_text else 'empty'}"
                )
                return ""

            self.logger.debug(f"Generated response: {generated_text[:200]}...")
            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"HuggingFace generation failed: {e}")
            return ""

    def cleanup_model(self):
        """Clean up model and free memory resources."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Delete pipeline components
            if hasattr(self.pipeline, 'model'):
                del self.pipeline.model
            if hasattr(self.pipeline, 'tokenizer'):
                del self.pipeline.tokenizer
            del self.pipeline
            self.pipeline = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache")
        except ImportError:
            pass

        self.logger.info("Model cleanup completed")


class OpenAIBackend(BaseModelInterface):
    """OpenAI API backend with Responses API and o1/o3 reasoning mode support."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key with proper fallback hierarchy: explicit > config > env
        self.api_key = (
            api_key
            or self.config.get("DEFAULT_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        # Check for deprecated models
        if model_name.startswith("gpt-4.1"):
            self.logger.warning(
                f"Model {model_name} was deprecated in August 2025. "
                "Consider using gpt-4o or gpt-5 for better performance."
            )

        # Determine if this is a reasoning model that doesn't support temperature
        self.is_reasoning_model = model_name.startswith(("o1", "o3", "gpt-5"))
        if self.is_reasoning_model and not self.reasoning_mode:
            self.reasoning_mode = True  # Auto-enable reasoning for o1/o3/gpt-5 models

        # Determine if this is a new generation model that supports Responses API
        self.supports_responses_api = model_name.startswith(("gpt-4.1", "gpt-5", "o1", "o3"))

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set DEFAULT_OPENAI_API_KEY in config, "
                "OPENAI_API_KEY environment variable, or pass api_key parameter"
            )

        # Validate API key format
        if not self.api_key.startswith("sk-"):
            self.logger.warning("OpenAI API key should start with 'sk-'")

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI client for model: {model_name}")
        except ImportError:
            # Fallback to requests-based approach for backward compatibility
            self.client = None
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.logger.warning("OpenAI client not available, falling back to requests")

    def _validate_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Validate and return supported parameters for the given model."""
        supported_params = {
            "supports_temperature": True,
            "supports_reasoning": False,
            "token_param": "max_tokens",
            "api_preference": "chat_completions"
        }

        if model_name.startswith(("o1", "o3", "o4")) and not model_name.startswith("o4-mini"):
            supported_params.update({
                "supports_temperature": False,
                "supports_reasoning": True,
                "token_param": "max_completion_tokens",
                "api_preference": "responses"
            })
        elif model_name.startswith("gpt-5"):
            supported_params.update({
                "supports_temperature": True,
                "supports_reasoning": True,
                "token_param": "max_tokens",
                "api_preference": "either"
            })

        return supported_params

    def _handle_openai_errors(self, error: Exception) -> None:
        """Handle OpenAI-specific errors with helpful messages."""
        error_str = str(error).lower()

        if "rate_limit_exceeded" in error_str or "quota" in error_str:
            self.logger.error(f"OpenAI quota/rate limit exceeded for {self.model_name}")
            raise RuntimeError(
                f"OpenAI account quota exceeded for {self.model_name}. "
                "Check billing and usage limits at https://platform.openai.com/account/usage"
            )

        if "unsupported parameter" in error_str and "temperature" in error_str:
            self.logger.error(f"Model {self.model_name} does not support temperature parameter")
            raise ValueError(
                f"Reasoning model {self.model_name} does not support temperature parameter. "
                "Temperature is automatically removed for reasoning models."
            )

        # Re-raise the original error if not handled
        raise error

    @retry_with_exponential_backoff(
        max_attempts=3, exceptions=(Exception,)
    )
    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using OpenAI API with Responses API and reasoning mode support."""
        try:
            # Use reasoning mode and effort overrides if provided
            use_reasoning = (
                reasoning_mode if reasoning_mode is not None else self.reasoning_mode
            )
            effort = reasoning_effort or self.reasoning_effort

            # Use modern OpenAI client if available and model supports it
            if self.client and self.supports_responses_api:
                try:
                    return self._generate_with_responses_api(
                        prompt, max_completion_tokens, max_tokens, use_reasoning, effort
                    )
                except Exception as e:
                    # If Responses API fails, try Chat Completions as fallback
                    self.logger.warning(
                        f"Responses API failed for {self.model_name}: {e}. "
                        "Falling back to Chat Completions API."
                    )
                    if self.client:
                        return self._generate_with_chat_completions(
                            prompt, max_completion_tokens, max_tokens, use_reasoning, effort
                        )
                    else:
                        raise  # Re-raise if no fallback available
            elif self.client:
                return self._generate_with_chat_completions(
                    prompt, max_completion_tokens, max_tokens, use_reasoning, effort
                )
            else:
                # Fallback to legacy requests-based approach
                return self._generate_with_requests(
                    prompt, max_completion_tokens, max_tokens, use_reasoning, effort
                )

        except Exception as e:
            # Use enhanced error handling
            self._handle_openai_errors(e)

    def _generate_with_responses_api(
        self,
        prompt: str,
        max_completion_tokens: int,
        max_tokens: int,
        use_reasoning: bool,
        effort: str,
    ) -> str:
        """Generate response using OpenAI Responses API."""
        try:
            # Build request parameters for Responses API based on documentation
            params = {
                "model": self.model_name,
                "input": prompt,  # Based on doc: "Text, image, or file inputs to the model"
            }

            # Add max_output_tokens - this parameter exists in the documentation
            if self.is_reasoning_model or use_reasoning:
                reasoning_tokens = self.get_reasoning_token_limit(effort)
                params["max_output_tokens"] = min(max_completion_tokens, reasoning_tokens)

                # Add reasoning configuration for models that support it (not o1)
                if not self.model_name.startswith("o1"):
                    effort_mapping = {"low": "minimal", "normal": "medium", "high": "high"}
                    params["reasoning"] = {
                        "effort": effort_mapping.get(effort, "medium")
                    }
                # Reasoning models don't support temperature parameter
            else:
                # For standard models: use max_output_tokens and temperature
                params["max_output_tokens"] = max_tokens
                params["temperature"] = self.temperature  # Doc shows temperature parameter exists

            self.logger.info(f"Using Responses API for model: {self.model_name}")
            response = self.client.responses.create(**params)

            # Extract token usage based on response format from documentation
            # Doc shows: "usage": {"input_tokens": 36, "output_tokens": 87, "total_tokens": 123}
            if hasattr(response, 'usage'):
                usage = response.usage
                self.last_token_usage = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            # Extract text based on response structure from documentation
            # Doc shows: response.output[0].content[0].text contains the actual text
            generated_text = ""
            if hasattr(response, 'output') and response.output and len(response.output) > 0:
                output_item = response.output[0]
                if hasattr(output_item, 'content') and output_item.content and len(output_item.content) > 0:
                    content_item = output_item.content[0]
                    if hasattr(content_item, 'text'):
                        generated_text = content_item.text

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"OpenAI Responses API failed: {e}")
            raise

    def _generate_with_chat_completions(
        self,
        prompt: str,
        max_completion_tokens: int,
        max_tokens: int,
        use_reasoning: bool,
        effort: str,
    ) -> str:
        """Generate response using OpenAI Chat Completions API."""
        try:
            # Build request parameters for Chat Completions API
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self.is_reasoning_model or use_reasoning:
                # For reasoning models: use max_completion_tokens and no temperature
                reasoning_tokens = self.get_reasoning_token_limit(effort)
                params["max_completion_tokens"] = min(
                    max_completion_tokens, reasoning_tokens
                )
                # Reasoning models don't support temperature parameter
            else:
                # For standard models: use max_tokens and temperature
                params["max_tokens"] = max_tokens
                params["temperature"] = self.temperature

            self.logger.info(f"Using Chat Completions API for model: {self.model_name}")
            response = self.client.chat.completions.create(**params)

            # Extract token usage information
            if hasattr(response, 'usage'):
                usage = response.usage
                self.last_token_usage = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            # Extract generated text
            generated_text = response.choices[0].message.content or ""
            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"OpenAI Chat Completions API failed: {e}")
            raise

    def _generate_with_requests(
        self,
        prompt: str,
        max_completion_tokens: int,
        max_tokens: int,
        use_reasoning: bool,
        effort: str,
    ) -> str:
        """Legacy requests-based generation for backward compatibility."""
        try:
            # Build payload based on reasoning mode
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self.is_reasoning_model or use_reasoning:
                # For reasoning models: use max_completion_tokens and no temperature
                reasoning_tokens = self.get_reasoning_token_limit(effort)
                payload["max_completion_tokens"] = min(
                    max_completion_tokens, reasoning_tokens
                )
                # Reasoning models don't support temperature parameter
            else:
                # For standard models: use max_tokens and temperature
                payload["max_tokens"] = max_tokens
                payload["temperature"] = self.temperature

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(self.api_url, json=payload, headers=headers)

            if response.status_code != 200:
                error_msg = (
                    f"OpenAI API Error: {response.status_code} - {response.text}"
                )
                self.logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Extract token usage information
            if "usage" in response_data:
                usage = response_data["usage"]
                self.last_token_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            generated_text = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            raise  # Let retry decorator handle it


class AnthropicBackend(BaseModelInterface):
    """Anthropic API backend with Claude thinking mode support."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key with proper fallback hierarchy: explicit > config > env
        self.api_key = (
            api_key
            or self.config.get("DEFAULT_ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )

        self.api_url = "https://api.anthropic.com/v1/messages"

        # All Claude models support thinking mode
        self.supports_thinking = True

        # Optional: Initialize official Anthropic client if available
        self.client = None
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.logger.info(f"Initialized Anthropic client for model: {model_name}")
        except ImportError:
            self.logger.info("Anthropic client not available, using requests")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set DEFAULT_ANTHROPIC_API_KEY in config, "
                "ANTHROPIC_API_KEY environment variable, or pass api_key parameter"
            )

    @retry_with_exponential_backoff(
        max_attempts=3, exceptions=(requests.exceptions.RequestException, Exception)
    )
    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using Anthropic API with thinking mode support."""
        try:
            # Use reasoning mode and effort overrides if provided
            use_reasoning = (
                reasoning_mode if reasoning_mode is not None else self.reasoning_mode
            )
            effort = reasoning_effort or self.reasoning_effort

            # Build payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }

            # Add thinking mode if reasoning is enabled - based on Anthropic docs
            if use_reasoning and self.supports_thinking:
                # According to the docs, thinking configuration should be:
                # "thinking": {"type": "enabled", "budget_tokens": X} format
                reasoning_tokens = self.get_reasoning_token_limit(effort)
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": reasoning_tokens,
                }
                # Note: Temperature is NOT compatible with thinking mode per Anthropic docs
                # Can only set top_p between 1 and 0.95 if needed
            else:
                payload["temperature"] = self.temperature

            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",  # Keep stable version for compatibility
            }

            response = requests.post(self.api_url, json=payload, headers=headers)

            if response.status_code != 200:
                error_msg = (
                    f"Anthropic API Error: {response.status_code} - {response.text}"
                )
                self.logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Extract token usage information
            if "usage" in response_data:
                usage = response_data["usage"]
                self.last_token_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            # Extract response content based on Anthropic documentation format
            # Doc shows: "content": [{"type": "text", "text": "Hi! My name is Claude."}]
            generated_parts = []
            for block in response_data.get("content", []):
                if block.get("type") == "text":
                    # Extract text from text blocks
                    generated_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    # For debugging, we might want to log thinking content
                    # Thinking blocks contain "text" field, not "content" field
                    thinking_text = block.get("text", "")
                    self.logger.debug(f"Claude thinking: {thinking_text[:200]}...")
                    # Don't include thinking text in final response per standard behavior

            generated_text = "\n".join(generated_parts)
            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Anthropic API request failed: {e}")
            raise  # Let retry decorator handle it
        except Exception as e:
            self.logger.error(f"Anthropic generation failed: {e}")
            raise  # Let retry decorator handle it


class OpenRouterBackend(BaseModelInterface):
    """OpenRouter API backend with model-specific optimization and advanced feature support."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key with proper fallback hierarchy: explicit > config > env
        self.api_key = (
            api_key
            or self.config.get("DEFAULT_OPENROUTER_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Detect model capabilities based on research findings
        self.model_capabilities = self._detect_model_capabilities()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set DEFAULT_OPENROUTER_API_KEY in config, "
                "OPENROUTER_API_KEY environment variable, or pass api_key parameter"
            )

    def _detect_model_capabilities(self) -> Dict[str, Any]:
        """Detect model-specific capabilities based on OpenRouter API research."""
        model_lower = self.model_name.lower()

        capabilities = {
            "supports_reasoning": False,
            "supports_tools": False,
            "has_rate_limits": False,
            "context_window": 4096,  # Default
            "optimal_temperature": self.temperature,
            "model_type": "standard",
            "max_completion_tokens": 8192,  # Default from research
        }

        # Reasoning models (DeepSeek R1, Qwen3) - Based on OpenRouter API docs
        if "deepseek" in model_lower and "r1" in model_lower:
            capabilities.update({
                "supports_reasoning": True,
                "context_window": 163840,  # 163K tokens confirmed
                "optimal_temperature": 0.1,
                "model_type": "reasoning",
                "reasoning_tokens": ["<think>", "</think>"],
                "max_completion_tokens": 8192,
                "supports_structured": True,  # DeepSeek R1 supports structured outputs
            })
        elif "qwen" in model_lower and ("qwen3" in model_lower or "30b" in model_lower):
            capabilities.update({
                "supports_reasoning": True,
                "context_window": 40960,  # 40K tokens confirmed
                "optimal_temperature": 0.1,
                "model_type": "reasoning",
                "reasoning_tokens": ["<think>", "</think>"],
                "default_stops": ["<|im_start|>", "<|im_end|>"],
                "max_completion_tokens": 4096,  # More conservative for Qwen3
            })

        # Tool calling models with enhanced capabilities
        elif "gemini" in model_lower and "2.0-flash" in model_lower:
            capabilities.update({
                "supports_tools": True,
                "supports_multimodal": True,  # Text and image inputs
                "context_window": 1048576,  # 1M+ tokens confirmed
                "optimal_temperature": 0.7,
                "model_type": "multimodal",
                "fast_ttft": True,
                "max_completion_tokens": 8192,
                "supports_structured": True,
            })
        elif "mistral" in model_lower and "small" in model_lower and "3.2" in model_lower:
            capabilities.update({
                "supports_tools": True,
                "supports_structured": True,  # Confirmed structured output support
                "supports_multimodal": True,  # Text and image inputs
                "context_window": 131072,  # 131K tokens confirmed
                "optimal_temperature": 0.7,
                "model_type": "structured",
                "max_completion_tokens": 8192,
            })
        elif "kimi" in model_lower and "k2" in model_lower:
            capabilities.update({
                "supports_tools": True,
                "context_window": 32768,  # 32K tokens confirmed
                "optimal_temperature": 0.7,
                "model_type": "moe",
                "max_completion_tokens": 4096,
                "active_params": "32B",  # 32B active, 1T total
                "supports_multipart": True,
            })

        # Rate limited model (LLaMA 3.3) - Enhanced with research findings
        elif "llama" in model_lower and "3.3" in model_lower:
            capabilities.update({
                "has_rate_limits": True,
                "rate_limit_delay": 65,  # 65 seconds to ensure we don't hit the 1 req/min limit
                "context_window": 65536,  # 65K tokens confirmed
                "optimal_temperature": 0.8,
                "model_type": "instruct",
                "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
                "max_completion_tokens": 4096,
                "multilingual": True,  # Supports multiple languages
                "supports_tools": True,  # LLaMA 3.3 supports tool calling
            })

        return capabilities

    def _get_model_specific_params(self, use_reasoning: bool, effort: str, max_tokens: int, **kwargs) -> Dict[str, Any]:
        """Get model-specific parameters based on OpenRouter API research."""
        # Use model's max_completion_tokens limit instead of arbitrary calculation
        max_allowed = self.model_capabilities.get("max_completion_tokens", 8192)
        effective_max_tokens = min(max_tokens, max_allowed)

        params = {
            "model": self.model_name,
            "max_tokens": effective_max_tokens,
            "temperature": self.model_capabilities["optimal_temperature"],
            "stream": False,  # Always disable streaming for reliability
        }

        # Reasoning models (DeepSeek R1, Qwen3) - Fixed for crossword evaluation compatibility
        if self.model_capabilities["supports_reasoning"] and use_reasoning:
            # DeepSeek R1 and Qwen3 use specific reasoning parameters
            if "deepseek" in self.model_name.lower() and "r1" in self.model_name.lower():
                params.update({
                    "reasoning": {"enabled": True, "include_in_response": False},  # Hide reasoning blocks
                    "temperature": 0.1,  # DeepSeek R1 works best with minimal temperature
                    "max_tokens": min(max_tokens, 4096),  # Reduce tokens since no reasoning blocks
                })
            elif "qwen" in self.model_name.lower() and ("qwen3" in self.model_name.lower() or "30b" in self.model_name.lower()):
                params.update({
                    "reasoning": {"enabled": True, "include_in_response": False},  # Hide reasoning blocks
                    "temperature": 0.1,  # Qwen3 optimal temperature
                    "max_tokens": min(max_tokens, 4096),  # Reduce tokens since no reasoning blocks
                })
                # Add default stops if not overridden
                if "default_stops" in self.model_capabilities:
                    params["stop"] = self.model_capabilities["default_stops"]
            else:
                # Generic reasoning model handling - disable reasoning for crossword compatibility
                params.update({
                    "temperature": 0.1,  # Use low temperature for consistent answers
                })

        # Tool calling models (Gemini, Mistral, Kimi K2)
        elif self.model_capabilities["supports_tools"]:
            # Add tool calling support if tools are provided
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
                params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Rate limited models (LLaMA 3.3) with enhanced parameter handling
        elif self.model_capabilities["has_rate_limits"]:
            # Add stop tokens for proper response termination
            if "stop_tokens" in self.model_capabilities:
                params["stop"] = self.model_capabilities["stop_tokens"]

            # LLaMA 3.3 specific optimizations
            if "llama" in self.model_name.lower() and "3.3" in self.model_name.lower():
                # Temperature optimization for LLaMA 3.3 instruct model
                params["temperature"] = 0.8  # Optimal for this model
                # Add tool calling support if tools are provided
                if "tools" in kwargs:
                    params["tools"] = kwargs["tools"]
                    params["tool_choice"] = kwargs.get("tool_choice", "auto")
            else:
                params["temperature"] = max(params["temperature"], 0.1)

        return params

    def _handle_rate_limiting(self):
        """Handle model-specific rate limiting with proper backoff strategy."""
        if self.model_capabilities.get("has_rate_limits", False):
            delay = self.model_capabilities.get("rate_limit_delay", 60)
            self.logger.info(
                f"Model {self.model_name} has rate limits (1 req/min). Waiting {delay}s between requests..."
            )
            time.sleep(delay)

        # Handle general free model rate limits (20 req/min for :free models)
        elif ":free" in self.model_name:
            # Conservative 3-second delay for free models to avoid hitting 20 req/min limit
            self.logger.debug(f"Free model {self.model_name} - applying 3s delay for rate limit compliance")
            time.sleep(3)

    def _post_process_response(self, generated_text: str, model_capabilities: Dict[str, Any]) -> str:
        """Post-process response to format for crossword parser compatibility."""
        if not generated_text:
            return generated_text

        # Handle reasoning tokens for DeepSeek R1 and Qwen3 (fallback in case include_in_response=False fails)
        if model_capabilities.get("reasoning_tokens"):
            think_start, think_end = model_capabilities["reasoning_tokens"]

            # Extract content after reasoning blocks
            if think_start in generated_text and think_end in generated_text:
                parts = generated_text.split(think_start)
                if len(parts) > 1:
                    reasoning_part = parts[1].split(think_end)[0]
                    self.logger.debug(f"Model reasoning detected: {reasoning_part[:100]}...")

                    # Extract final answer after reasoning
                    if think_end in generated_text:
                        final_parts = generated_text.split(think_end)
                        if len(final_parts) > 1:
                            generated_text = final_parts[-1].strip()

        # Handle stop tokens for LLaMA models
        if model_capabilities.get("stop_tokens"):
            for stop_token in model_capabilities["stop_tokens"]:
                if stop_token in generated_text:
                    generated_text = generated_text.split(stop_token)[0]

        # Additional crossword-specific response normalization
        generated_text = self._normalize_crossword_response(generated_text, model_capabilities)

        return generated_text.strip()

    def _normalize_crossword_response(self, response: str, model_capabilities: Dict[str, Any]) -> str:
        """Normalize response format for crossword parser compatibility."""
        if not response:
            return response

        # Remove common conversational markers
        conversation_markers = [
            "Here is my response:",
            "Here's my answer:",
            "The answer is:",
            "Answer:",
            "Response:",
            "My answer:",
            "Solution:",
            "Here's the solution:",
            "Based on the clue,",
            "Looking at this clue,",
            "For this crossword clue,",
        ]

        for marker in conversation_markers:
            if marker.lower() in response.lower():
                # Split on marker and take content after it
                parts = response.lower().split(marker.lower())
                if len(parts) > 1:
                    response = parts[1].strip()
                    break

        # Remove common reasoning/explanation patterns
        explanation_patterns = [
            "because", "since", "as", "therefore", "thus", "so", "hence",
            "this is", "it means", "which refers to", "referring to"
        ]

        # Split on explanation patterns and keep only the first part (likely the answer)
        for pattern in explanation_patterns:
            if pattern in response.lower():
                parts = response.lower().split(pattern)
                if len(parts) > 1 and len(parts[0].strip()) > 0:
                    response = parts[0].strip()
                    break

        # Handle model-specific formatting issues
        model_name_lower = self.model_name.lower()

        # Gemini 2.0 specific - often returns structured responses
        if "gemini" in model_name_lower:
            # Remove markdown formatting
            response = response.replace("**", "").replace("*", "").replace("#", "")
            # Extract from numbered lists
            if response.startswith(("1.", "1)", "- ")):
                lines = response.split('\n')
                if lines:
                    first_line = lines[0]
                    # Remove numbering
                    response = first_line.replace("1.", "").replace("1)", "").replace("- ", "").strip()

        # Mistral/LLaMA specific - often includes explanations
        elif any(x in model_name_lower for x in ["mistral", "llama"]):
            # Split on punctuation that might separate answer from explanation
            for separator in [".", ",", ";", ":", "!", "?"]:
                if separator in response:
                    parts = response.split(separator)
                    if len(parts[0].strip()) > 0:
                        response = parts[0].strip()
                        break

        # Final cleanup - remove extra whitespace and punctuation
        response = response.strip().strip('.,!?;:"\'')

        # Extract single words if response contains multiple words (common in crosswords)
        words = response.split()
        if len(words) == 1:
            response = words[0]
        elif len(words) > 1:
            # For crosswords, often the answer is the first word or a compound word
            # Keep the response as is for now - let the parser handle it
            pass

        return response

    @retry_with_exponential_backoff(
        max_attempts=3, exceptions=(requests.exceptions.RequestException, Exception)
    )
    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate response using OpenRouter API with model-specific optimization."""
        try:
            # Handle rate limiting before making request
            self._handle_rate_limiting()

            # Use reasoning mode and effort overrides if provided
            use_reasoning = (
                reasoning_mode if reasoning_mode is not None else self.reasoning_mode
            )
            effort = reasoning_effort or self.reasoning_effort

            # Get model-specific optimized parameters
            payload = self._get_model_specific_params(
                use_reasoning, effort, max_tokens, **kwargs
            )

            # Add standard message format
            payload["messages"] = [{"role": "user", "content": prompt}]

            # Log model capabilities for debugging
            self.logger.info(
                f"Using {self.model_name} with capabilities: "
                f"reasoning={self.model_capabilities.get('supports_reasoning')}, "
                f"tools={self.model_capabilities.get('supports_tools')}, "
                f"type={self.model_capabilities.get('model_type')}"
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/buddhi-pragati/benchmark",
                "X-Title": "Buddhi-Pragati Crossword Benchmark",
            }

            self.logger.debug(
                f"Sending request to OpenRouter for model: {self.model_name}"
            )
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=60
            )

            if response.status_code != 200:
                error_msg = (
                    f"OpenRouter API Error: {response.status_code} - {response.text}"
                )
                self.logger.error(error_msg)

                # Parse error for specific issues
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_detail = error_data["error"].get(
                            "message", "Unknown error"
                        )
                        self.logger.error(f"OpenRouter error detail: {error_detail}")

                        # Enhanced rate limiting handling
                        if response.status_code == 429:
                            self.logger.warning(
                                f"Rate limited by OpenRouter for {self.model_name}, implementing backoff strategy..."
                            )
                            # Implement exponential backoff for rate limits
                            if ":free" in self.model_name:
                                # Free models have 20 req/min limit - wait longer
                                backoff_delay = 10
                            elif self.model_capabilities.get("has_rate_limits"):
                                # Models like LLaMA 3.3 with 1 req/min - wait full minute
                                backoff_delay = 65
                            else:
                                # Default backoff
                                backoff_delay = 5

                            self.logger.info(f"Waiting {backoff_delay}s for rate limit backoff...")
                            time.sleep(backoff_delay)
                except Exception:
                    pass

                raise Exception(error_msg)

            response_data = response.json()

            # Extract token usage if available
            if "usage" in response_data:
                usage = response_data["usage"]
                self.last_token_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            # Enhanced response extraction with comprehensive error handling
            choices = response_data.get("choices", [])
            if not choices:
                # Check for specific error conditions that cause empty choices
                if "error" in response_data:
                    error_info = response_data["error"]
                    error_code = error_info.get("code", 0)
                    error_msg = error_info.get("message", "Unknown error")

                    # Handle specific error types
                    if error_code == 403:  # Content moderation
                        self.logger.error(f"Content flagged by moderation for {self.model_name}: {error_msg}")
                        raise Exception(f"Content moderation error: {error_msg}")
                    elif error_code == 429:  # Rate limit
                        self.logger.error(f"Rate limit exceeded for {self.model_name}: {error_msg}")
                        raise Exception(f"Rate limit exceeded: {error_msg}")
                    elif error_code == 502:  # Model down
                        self.logger.error(f"Model {self.model_name} is down: {error_msg}")
                        raise Exception(f"Model unavailable: {error_msg}")
                    elif error_code == 503:  # No providers
                        self.logger.error(f"No providers available for {self.model_name}: {error_msg}")
                        raise Exception(f"No providers available: {error_msg}")
                    else:
                        self.logger.error(f"API error {error_code} for {self.model_name}: {error_msg}")
                        raise Exception(f"API error {error_code}: {error_msg}")

                self.logger.warning(f"No choices in OpenRouter response for {self.model_name} - possible model warmup")
                raise Exception(f"Empty response from {self.model_name} - model may be warming up")

            choice = choices[0]
            generated_text = ""

            # Extract content from choice with fallbacks
            if "message" in choice and choice["message"]:
                generated_text = choice["message"].get("content", "")
                # For reasoning models, if content is empty but reasoning exists,
                # treat it as unsatisfactory response (not empty)
                if not generated_text and "reasoning" in choice["message"]:
                    reasoning_text = choice["message"].get("reasoning", "")
                    if reasoning_text:
                        self.logger.info(f"No content field but found reasoning for {self.model_name}")
                        self.logger.debug(f"Reasoning content: {reasoning_text[:200]}...")
                        # Return the reasoning as the response - let parser handle it
                        generated_text = reasoning_text
            elif "delta" in choice and choice["delta"]:  # Handle streaming responses
                generated_text = choice["delta"].get("content", "")
            elif "text" in choice:  # Some models use direct text field
                generated_text = choice.get("text", "")

            # Check for finish_reason indicating issues
            finish_reason = choice.get("finish_reason")
            if finish_reason == "content_filter":
                self.logger.error(f"Content filtered for {self.model_name}")
                raise Exception("Response blocked by content filter")
            elif finish_reason == "error":
                self.logger.error(f"Model error for {self.model_name}")
                raise Exception("Model returned error finish reason")

            # Apply model-specific post-processing
            processed_text = self._post_process_response(generated_text, self.model_capabilities)

            # Phase 1: Check if response is truly empty (no content at all)
            if not processed_text or not processed_text.strip():
                self.logger.error(
                    f"Truly empty response from {self.model_name} - model returned no content. "
                    f"Raw response: {str(response_data)[:200]}... "
                    f"Finish reason: {finish_reason}"
                )
                raise Exception(f"Empty response from {self.model_name}")

            # Phase 2: Check if response quality is satisfactory for crossword parsing
            # Even unsatisfactory responses should be passed through for parser evaluation
            if not validate_response(processed_text, min_length=5):
                self.logger.info(
                    f"Unsatisfactory response format from {self.model_name} (may not be crossword format): '{processed_text[:100]}...'"
                )
                # Pass through unsatisfactory responses - let parser determine usability
                # This handles cases like reasoning text, partial responses, etc.

            # Classify and log response type for clarity
            is_satisfactory = validate_response(processed_text, min_length=5)
            response_type = "satisfactory" if is_satisfactory else "unsatisfactory"

            self.logger.info(f"Returning {response_type} response from {self.model_name} ({len(processed_text)} chars): '{processed_text[:100]}...'")
            return processed_text.strip()

        except requests.exceptions.Timeout:
            self.logger.error("OpenRouter API request timed out")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            raise  # Let retry decorator handle it
        except Exception as e:
            self.logger.error(f"OpenRouter generation failed: {e}")
            raise  # Let retry decorator handle it


class SarvamAIBackend(BaseModelInterface):
    """SarvamAI API backend with official client library and reasoning effort support for Indic languages."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key with proper fallback hierarchy - note the correct parameter name
        # Based on docs: api_subscription_key not api_key
        self.api_subscription_key = (
            api_key
            or self.config.get("DEFAULT_SARVAM_API_SUBSCRIPTION_KEY")
            or self.config.get("DEFAULT_SARVAM_API_KEY")  # Backward compatibility
            or os.environ.get("SARVAM_API_SUBSCRIPTION_KEY")
            or os.environ.get("SARVAM_API_KEY")  # Backward compatibility
        )

        # Fixed model name based on documentation - only "sarvam-m" is supported
        self.api_model_name = "sarvam-m"

        # Initialize SarvamAI client
        self.client = None
        try:
            from sarvamai import SarvamAI
            self.client = SarvamAI(api_subscription_key=self.api_subscription_key)
            self.logger.info(f"Initialized SarvamAI client for model: {model_name}")
        except ImportError:
            # Fallback to requests-based approach for backward compatibility
            self.client = None
            self.api_url = "https://api.sarvam.ai/v1/chat/completions"
            self.logger.warning("SarvamAI client not available, falling back to requests")

        if not self.api_subscription_key:
            raise ValueError(
                "SarvamAI API subscription key is required. Set DEFAULT_SARVAM_API_SUBSCRIPTION_KEY in config, "
                "SARVAM_API_SUBSCRIPTION_KEY environment variable, or pass api_key parameter"
            )

    @retry_with_exponential_backoff(
        max_attempts=3, exceptions=(Exception,)
    )
    def generate_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000,
        max_tokens: int = 1000,
        reasoning_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using SarvamAI API with reasoning effort support."""
        try:
            # Use reasoning mode and effort overrides if provided
            use_reasoning = (
                reasoning_mode if reasoning_mode is not None else self.reasoning_mode
            )
            effort = reasoning_effort or self.reasoning_effort

            # Use official client if available, otherwise fall back to requests
            if self.client:
                return self._generate_with_client(prompt, max_tokens, use_reasoning, effort)
            else:
                return self._generate_with_requests(prompt, max_tokens, use_reasoning, effort)

        except Exception as e:
            self.logger.error(f"SarvamAI generation failed: {e}")
            raise  # Let retry decorator handle it

    def _generate_with_client(
        self,
        prompt: str,
        max_tokens: int,
        use_reasoning: bool,
        effort: str,
    ) -> str:
        """Generate response using SarvamAI official client."""
        try:
            # Build parameters based on SarvamAI client documentation
            # Note: SarvamAI client doesn't need model parameter, it's implicit
            params = {
                "messages": [{"content": prompt, "role": "user"}],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            # Add reasoning effort if enabled - map to SarvamAI format
            if use_reasoning:
                # Based on docs: reasoning_effort: "low", "medium", "high"
                effort_mapping = {"low": "low", "normal": "medium", "high": "high"}
                params["reasoning_effort"] = effort_mapping.get(effort, "medium")

            self.logger.info(f"Using SarvamAI client for model: {self.api_model_name}")
            response = self.client.chat.completions(**params)

            # Extract token usage based on documentation format
            # Doc shows: "usage": {"completion_tokens": 42, "prompt_tokens": 42, "total_tokens": 42}
            if hasattr(response, 'usage'):
                usage = response.usage
                self.last_token_usage = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            # Extract generated text based on documentation format
            # Doc shows: response.choices[0].message.content
            generated_text = ""
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    generated_text = choice.message.content or ""

            # Validate response
            if not validate_response(generated_text):
                self.logger.warning(
                    f"Invalid response from SarvamAI: {generated_text[:100] if generated_text else 'empty'}"
                )
                return ""

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"SarvamAI client generation failed: {e}")
            raise

    def _generate_with_requests(
        self,
        prompt: str,
        max_tokens: int,
        use_reasoning: bool,
        effort: str,
    ) -> str:
        """Legacy requests-based generation for backward compatibility."""
        try:
            # Build payload with the correct model name
            payload = {
                "model": self.api_model_name,  # Always "sarvam-m"
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            # Add reasoning effort if enabled
            if use_reasoning:
                # Map our reasoning effort to SarvamAI's levels
                effort_mapping = {"low": "low", "normal": "medium", "high": "high"}
                payload["reasoning_effort"] = effort_mapping.get(effort, "medium")

            headers = {
                "api-subscription-key": self.api_subscription_key,  # Correct header name
                "Content-Type": "application/json",
            }

            self.logger.debug(
                f"Sending request to SarvamAI with model: {self.api_model_name}"
            )
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=30
            )

            if response.status_code != 200:
                error_msg = (
                    f"SarvamAI API Error: {response.status_code} - {response.text}"
                )
                self.logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Extract token usage if available
            if "usage" in response_data:
                usage = response_data["usage"]
                self.last_token_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "reasoning_mode": use_reasoning,
                    "reasoning_effort": effort,
                }

            generated_text = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Validate response
            if not validate_response(generated_text):
                self.logger.warning(
                    f"Invalid response from SarvamAI: {generated_text[:100] if generated_text else 'empty'}"
                )
                return ""

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"SarvamAI API request failed: {e}")
            raise  # Let retry decorator handle it
