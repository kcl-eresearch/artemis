"""Custom exceptions used across the Artemis service.

This module defines the exception hierarchy for Artemis. All custom exceptions
derive from ArtemisError, with UpstreamServiceError being the primary exception
used when external services (SearXNG, LLM backend) fail or return invalid responses.

The UpstreamServiceError is mapped to HTTP 502 (Bad Gateway) by default, indicating
that the upstream service failed to fulfill a valid request.
"""


class ArtemisError(Exception):
    """Base exception for Artemis-specific failures.

    All custom exceptions in Artemis should inherit from this class to allow
    for centralized exception handling and to distinguish service-specific
    errors from generic Python exceptions.
    """


class UpstreamServiceError(ArtemisError):
    """Raised when an upstream dependency fails or returns invalid data.

    This exception is used when calls to external services (SearXNG search API,
    LiteLLM-compatible LLM endpoints) fail for any reason, including:
    - Network timeouts
    - HTTP error responses (4xx, 5xx)
    - Invalid/malformed responses
    - Service unavailability

    The default HTTP status code is 502 (Bad Gateway), which indicates that
    the gateway/proxy received an invalid response from the upstream server.

    Attributes:
        message: Human-readable error description
        status_code: HTTP status code to return to the client (default: 502)
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
