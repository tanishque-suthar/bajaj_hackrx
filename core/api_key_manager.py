import os
import time
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages multiple API keys with automatic fallback and rate limit handling"""
    
    def __init__(self, service_name: str, env_var_name: str, rate_limit_cooldown: int = 120):
        """
        Initialize API Key Manager
        
        Args:
            service_name: Name of the service (e.g., "HuggingFace", "Gemini")
            env_var_name: Environment variable name containing comma-separated API keys
            rate_limit_cooldown: Seconds to wait before retrying rate-limited key (default: 2 minutes)
        """
        self.service_name = service_name
        self.rate_limit_cooldown = rate_limit_cooldown
        
        # Load API keys from environment variable
        api_keys_str = os.getenv(env_var_name, "")
        if not api_keys_str:
            raise ValueError(f"{env_var_name} not found in environment variables")
        
        # Parse comma-separated keys and remove empty strings
        self.api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
        
        if not self.api_keys:
            raise ValueError(f"No valid API keys found in {env_var_name}")
        
        logger.info(f"Initialized {self.service_name} APIKeyManager with {len(self.api_keys)} keys")
        
        # Track key states
        self.current_key_index = 0
        self.rate_limited_keys: Dict[str, datetime] = {}  # key -> when rate limit expires
        self.failed_keys: set = set()  # permanently failed keys
        
        # Statistics for monitoring
        self.key_usage_count: Dict[str, int] = {key: 0 for key in self.api_keys}
        self.key_success_count: Dict[str, int] = {key: 0 for key in self.api_keys}
        self.key_failure_count: Dict[str, int] = {key: 0 for key in self.api_keys}
    
    def get_next_available_key(self) -> Optional[str]:
        """
        Get the next available API key using round-robin strategy
        
        Returns:
            Next available API key or None if all keys are unavailable
        """
        current_time = datetime.now()
        
        # Clean up expired rate limits
        self._cleanup_expired_rate_limits()
        
        # Get list of available keys (not failed and not rate limited)
        available_keys = self._get_available_keys()
        
        if not available_keys:
            logger.error(f"No available {self.service_name} API keys")
            return None
        
        # Round-robin selection from available keys
        key = available_keys[self.current_key_index % len(available_keys)]
        self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        
        # Update usage statistics
        self.key_usage_count[key] += 1
        
        logger.debug(f"Selected {self.service_name} API key: {self._mask_key(key)}")
        return key
    
    def mark_key_rate_limited(self, api_key: str, custom_cooldown: Optional[int] = None):
        """
        Mark an API key as rate limited
        
        Args:
            api_key: The API key that hit rate limit
            custom_cooldown: Optional custom cooldown period in seconds
        """
        cooldown = custom_cooldown or self.rate_limit_cooldown
        expire_time = datetime.now() + timedelta(seconds=cooldown)
        self.rate_limited_keys[api_key] = expire_time
        
        self.key_failure_count[api_key] += 1
        
        logger.warning(
            f"Marked {self.service_name} API key as rate limited for {cooldown}s: {self._mask_key(api_key)}"
        )
    
    def mark_key_failed(self, api_key: str):
        """
        Mark an API key as permanently failed (e.g., invalid credentials)
        
        Args:
            api_key: The API key that failed permanently
        """
        self.failed_keys.add(api_key)
        self.key_failure_count[api_key] += 1
        
        logger.error(f"Marked {self.service_name} API key as permanently failed: {self._mask_key(api_key)}")
    
    def mark_key_successful(self, api_key: str):
        """
        Mark an API key as successful (for statistics)
        
        Args:
            api_key: The API key that succeeded
        """
        self.key_success_count[api_key] += 1
    
    def get_statistics(self) -> Dict:
        """Get usage statistics for all API keys"""
        available_count = len(self._get_available_keys())
        rate_limited_count = len(self.rate_limited_keys)
        failed_count = len(self.failed_keys)
        
        return {
            "service": self.service_name,
            "total_keys": len(self.api_keys),
            "available_keys": available_count,
            "rate_limited_keys": rate_limited_count,
            "failed_keys": failed_count,
            "key_usage": {self._mask_key(k): v for k, v in self.key_usage_count.items()},
            "key_success": {self._mask_key(k): v for k, v in self.key_success_count.items()},
            "key_failures": {self._mask_key(k): v for k, v in self.key_failure_count.items()},
        }
    
    def reset_key_state(self, api_key: str):
        """
        Reset the state of a specific API key (remove from failed/rate-limited)
        
        Args:
            api_key: The API key to reset
        """
        if api_key in self.failed_keys:
            self.failed_keys.remove(api_key)
            logger.info(f"Reset failed state for {self.service_name} key: {self._mask_key(api_key)}")
        
        if api_key in self.rate_limited_keys:
            del self.rate_limited_keys[api_key]
            logger.info(f"Reset rate limit for {self.service_name} key: {self._mask_key(api_key)}")
    
    def _cleanup_expired_rate_limits(self):
        """Remove expired rate limits"""
        current_time = datetime.now()
        expired_keys = [
            key for key, expire_time in self.rate_limited_keys.items()
            if current_time >= expire_time
        ]
        
        for key in expired_keys:
            del self.rate_limited_keys[key]
            logger.info(f"Rate limit expired for {self.service_name} key: {self._mask_key(key)}")
    
    def _get_available_keys(self) -> List[str]:
        """Get list of currently available API keys"""
        available = []
        for key in self.api_keys:
            if key not in self.failed_keys and key not in self.rate_limited_keys:
                available.append(key)
        return available
    
    def _mask_key(self, api_key: str) -> str:
        """Mask API key for logging (show only first 8 and last 4 characters)"""
        if len(api_key) <= 12:
            return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
        return f"{api_key[:8]}{'*' * (len(api_key) - 12)}{api_key[-4:]}"


class HuggingFaceAPIKeyManager(APIKeyManager):
    """Specialized API Key Manager for Hugging Face"""
    
    def __init__(self):
        super().__init__(
            service_name="HuggingFace",
            env_var_name="HF_API_TOKENS",
            rate_limit_cooldown=120  # 2 minutes
        )
    
    def is_rate_limit_error(self, error_message: str) -> bool:
        """
        Check if error message indicates rate limiting
        
        Args:
            error_message: Error message from API response
            
        Returns:
            True if error indicates rate limiting
        """
        error_lower = error_message.lower()
        rate_limit_indicators = [
            'rate limit', 'rate_limit', 'rate-limit',
            'quota exceeded', 'quota_exceeded',
            'too many requests', 'too_many_requests',
            'throttle', 'throttled',
            '429', 'http 429'
        ]
        
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def is_auth_error(self, error_message: str) -> bool:
        """
        Check if error message indicates authentication failure
        
        Args:
            error_message: Error message from API response
            
        Returns:
            True if error indicates authentication failure
        """
        error_lower = error_message.lower()
        auth_error_indicators = [
            'unauthorized', 'invalid token', 'invalid_token',
            'api key', 'api_key', 'authentication',
            'forbidden', '401', 'http 401', '403', 'http 403'
        ]
        
        return any(indicator in error_lower for indicator in auth_error_indicators)


class GeminiAPIKeyManager(APIKeyManager):
    """Specialized API Key Manager for Gemini API"""
    
    def __init__(self):
        super().__init__(
            service_name="Gemini",
            env_var_name="GEMINI_API_TOKENS", 
            rate_limit_cooldown=120  # 2 minutes
        )
    
    def is_rate_limit_error(self, error_message: str) -> bool:
        """Check if error message indicates rate limiting for Gemini API"""
        error_lower = error_message.lower()
        rate_limit_indicators = [
            'quota exceeded', 'quotaexceeded',
            'rate limit', 'rate_limit', 'rate-limit',
            'requests per minute', 'requests_per_minute',
            'resource_exhausted', 'resource exhausted',
            '429', 'http 429'
        ]
        
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def is_auth_error(self, error_message: str) -> bool:
        """Check if error message indicates authentication failure for Gemini API"""
        error_lower = error_message.lower()
        auth_error_indicators = [
            'api key not valid', 'api_key_not_valid',
            'invalid argument', 'invalid_argument',
            'permission denied', 'permission_denied',
            'unauthenticated', '401', 'http 401',
            '403', 'http 403'
        ]
        
        return any(indicator in error_lower for indicator in auth_error_indicators)


class PineconeAPIKeyManager(APIKeyManager):
    """Specialized API Key Manager for Pinecone"""
    
    def __init__(self):
        super().__init__(
            service_name="Pinecone",
            env_var_name="PINECONE_API_KEY",
            rate_limit_cooldown=120  # 2 minutes
        )
    
    def is_rate_limit_error(self, error_message: str) -> bool:
        """Check if error message indicates rate limiting for Pinecone API"""
        error_lower = error_message.lower()
        rate_limit_indicators = [
            'rate limit', 'rate_limit', 'rate-limit',
            'quota exceeded', 'quota_exceeded',
            'too many requests', 'too_many_requests',
            'throttle', 'throttled',
            '429', 'http 429',
            'requests per second', 'requests_per_second'
        ]
        
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def is_auth_error(self, error_message: str) -> bool:
        """Check if error message indicates authentication failure for Pinecone API"""
        error_lower = error_message.lower()
        auth_error_indicators = [
            'unauthorized', 'invalid api key', 'invalid_api_key',
            'authentication failed', 'authentication_failed',
            'forbidden', '401', 'http 401',
            '403', 'http 403',
            'api key not found', 'api_key_not_found'
        ]
        
        return any(indicator in error_lower for indicator in auth_error_indicators)


class ModelManager:
    """Manages model fallback for Gemini API"""
    
    def __init__(self, models: List[str], rate_limit_cooldown: int = 120):
        """
        Initialize Model Manager
        
        Args:
            models: List of model names in priority order
            rate_limit_cooldown: Seconds to wait before retrying rate-limited model
        """
        self.models = models
        self.rate_limit_cooldown = rate_limit_cooldown
        self.current_model_index = 0
        
        # Track model states
        self.rate_limited_models: Dict[str, datetime] = {}  # model -> when rate limit expires
        self.failed_models: set = set()  # permanently failed models
        
        # Statistics for monitoring
        self.model_usage_count: Dict[str, int] = {model: 0 for model in self.models}
        self.model_success_count: Dict[str, int] = {model: 0 for model in self.models}
        self.model_failure_count: Dict[str, int] = {model: 0 for model in self.models}
        
        logger.info(f"Initialized ModelManager with models: {', '.join(self.models)}")
    
    def get_next_available_model(self) -> Optional[str]:
        """
        Get the next available model using priority order
        
        Returns:
            Next available model or None if all models are unavailable
        """
        self._cleanup_expired_rate_limits()
        available_models = self._get_available_models()
        
        if not available_models:
            logger.error("No available models")
            return None
        
        # Return the highest priority available model
        model = available_models[0]
        self.model_usage_count[model] += 1
        
        logger.debug(f"Selected model: {model}")
        return model
    
    def mark_model_rate_limited(self, model: str, custom_cooldown: Optional[int] = None):
        """Mark a model as rate limited"""
        cooldown = custom_cooldown or self.rate_limit_cooldown
        expire_time = datetime.now() + timedelta(seconds=cooldown)
        self.rate_limited_models[model] = expire_time
        self.model_failure_count[model] += 1
        
        logger.warning(f"Marked model {model} as rate limited for {cooldown}s")
    
    def mark_model_failed(self, model: str):
        """Mark a model as permanently failed"""
        self.failed_models.add(model)
        self.model_failure_count[model] += 1
        
        logger.error(f"Marked model {model} as permanently failed")
    
    def mark_model_successful(self, model: str):
        """Mark a model as successful"""
        self.model_success_count[model] += 1
    
    def get_statistics(self) -> Dict:
        """Get usage statistics for all models"""
        available_count = len(self._get_available_models())
        rate_limited_count = len(self.rate_limited_models)
        failed_count = len(self.failed_models)
        
        return {
            "total_models": len(self.models),
            "available_models": available_count,
            "rate_limited_models": rate_limited_count,
            "failed_models": failed_count,
            "model_usage": self.model_usage_count.copy(),
            "model_success": self.model_success_count.copy(),
            "model_failures": self.model_failure_count.copy(),
        }
    
    def reset_model_state(self, model: str):
        """Reset the state of a specific model"""
        if model in self.failed_models:
            self.failed_models.remove(model)
            logger.info(f"Reset failed state for model: {model}")
        
        if model in self.rate_limited_models:
            del self.rate_limited_models[model]
            logger.info(f"Reset rate limit for model: {model}")
    
    def _cleanup_expired_rate_limits(self):
        """Remove expired model rate limits"""
        current_time = datetime.now()
        expired_models = [
            model for model, expire_time in self.rate_limited_models.items()
            if current_time >= expire_time
        ]
        
        for model in expired_models:
            del self.rate_limited_models[model]
            logger.info(f"Rate limit expired for model: {model}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of currently available models in priority order"""
        available = []
        for model in self.models:  # Maintain priority order
            if model not in self.failed_models and model not in self.rate_limited_models:
                available.append(model)
        return available
