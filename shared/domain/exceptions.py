"""
Rich Domain Exceptions

Comprehensive exception hierarchy for domain-specific errors.
Supports structured error information, error codes, and context.
"""

from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ErrorCode(str, Enum):
    """Standard error codes for domain exceptions."""

    # Domain errors
    DOMAIN_VALIDATION_ERROR = "DOMAIN_VALIDATION_ERROR"
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    ENTITY_NOT_FOUND = "ENTITY_NOT_FOUND"
    ENTITY_ALREADY_EXISTS = "ENTITY_ALREADY_EXISTS"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"

    # Enrollment errors
    ENROLLMENT_POLICY_VIOLATION = "ENROLLMENT_POLICY_VIOLATION"
    ENROLLMENT_QUOTA_EXCEEDED = "ENROLLMENT_QUOTA_EXCEEDED"
    PREREQUISITE_NOT_MET = "PREREQUISITE_NOT_MET"
    SCHEDULE_CONFLICT = "SCHEDULE_CONFLICT"

    # Authentication & Authorization
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # External service errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"

    # ML Service errors
    ML_SERVICE_UNAVAILABLE = "ML_SERVICE_UNAVAILABLE"
    ML_MODEL_NOT_LOADED = "ML_MODEL_NOT_LOADED"
    ML_PREDICTION_FAILED = "ML_PREDICTION_FAILED"

    # Data errors
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"
    ENCRYPTION_ERROR = "ENCRYPTION_ERROR"
    DECRYPTION_ERROR = "DECRYPTION_ERROR"

    # System errors
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    EVENT_STORE_ERROR = "EVENT_STORE_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class DomainException(Exception):
    """
    Base class for all domain exceptions.

    Provides structured error information with error codes, context, and metadata.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = 500,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize domain exception.

        Args:
            message: Human-readable error message
            error_code: Standard error code
            status_code: HTTP status code (default: 500)
            context: Additional context data
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.context = context or {}
        self.cause = cause

        # Log the exception
        logger.error(
            "Domain exception raised",
            error_code=error_code.value,
            message=message,
            status_code=status_code,
            context=context,
            exception_type=type(self).__name__,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.error_code.value,
            "message": self.message,
            "type": type(self).__name__,
        }
        if self.context:
            result["context"] = self.context
        return result


class ValidationError(DomainException):
    """Raised when domain validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)

        super().__init__(
            message=message,
            error_code=ErrorCode.DOMAIN_VALIDATION_ERROR,
            status_code=400,
            context=context,
            **kwargs
        )


class BusinessRuleViolationError(DomainException):
    """Raised when a business rule is violated."""

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if rule_name:
            context["rule_name"] = rule_name

        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_RULE_VIOLATION,
            status_code=400,
            context=context,
            **kwargs
        )


class EntityNotFoundError(DomainException):
    """Raised when an entity is not found."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str | None = None,
        **kwargs
    ):
        message = kwargs.pop("message", None) or f"{entity_type} not found"
        if entity_id:
            message += f" (ID: {entity_id})"

        context = kwargs.pop("context", {})
        context["entity_type"] = entity_type
        if entity_id:
            context["entity_id"] = entity_id

        super().__init__(
            message=message,
            error_code=ErrorCode.ENTITY_NOT_FOUND,
            status_code=404,
            context=context,
            **kwargs
        )


class EntityAlreadyExistsError(DomainException):
    """Raised when attempting to create an entity that already exists."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str | None = None,
        **kwargs
    ):
        message = kwargs.pop("message", None) or f"{entity_type} already exists"
        if entity_id:
            message += f" (ID: {entity_id})"

        context = kwargs.pop("context", {})
        context["entity_type"] = entity_type
        if entity_id:
            context["entity_id"] = entity_id

        super().__init__(
            message=message,
            error_code=ErrorCode.ENTITY_ALREADY_EXISTS,
            status_code=409,
            context=context,
            **kwargs
        )


class EnrollmentPolicyViolationError(DomainException):
    """Raised when enrollment policies are violated."""

    def __init__(
        self,
        reason: str,
        violated_rules: list[str] | None = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if violated_rules:
            context["violated_rules"] = violated_rules

        super().__init__(
            message=reason,
            error_code=ErrorCode.ENROLLMENT_POLICY_VIOLATION,
            status_code=400,
            context=context,
            **kwargs
        )
        self.reason = reason
        self.violated_rules = violated_rules or []


class ExternalServiceError(DomainException):
    """Raised when an external service call fails."""

    def __init__(
        self,
        service_name: str,
        message: str | None = None,
        timeout: bool = False,
        **kwargs
    ):
        error_code = ErrorCode.EXTERNAL_SERVICE_TIMEOUT if timeout else ErrorCode.EXTERNAL_SERVICE_ERROR
        default_message = f"{service_name} service {'timed out' if timeout else 'returned an error'}"

        context = kwargs.pop("context", {})
        context["service_name"] = service_name
        context["timeout"] = timeout

        super().__init__(
            message=message or default_message,
            error_code=error_code,
            status_code=503,
            context=context,
            **kwargs
        )
        self.service_name = service_name
        self.timeout = timeout


class CircuitBreakerOpenError(DomainException):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(
        self,
        service_name: str,
        **kwargs
    ):
        message = f"Circuit breaker is open for {service_name}. Service is temporarily unavailable."

        context = kwargs.pop("context", {})
        context["service_name"] = service_name
        context["circuit_state"] = "open"

        super().__init__(
            message=message,
            error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            status_code=503,
            context=context,
            **kwargs
        )
        self.service_name = service_name


class MLServiceUnavailableError(DomainException):
    """Raised when ML service is unavailable and fallback should be used."""

    def __init__(
        self,
        reason: str = "ML service unavailable",
        fallback_used: bool = False,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context["fallback_used"] = fallback_used

        super().__init__(
            message=reason,
            error_code=ErrorCode.ML_SERVICE_UNAVAILABLE,
            status_code=503,
            context=context,
            **kwargs
        )
        self.fallback_used = fallback_used


class DatabaseError(DomainException):
    """Raised when a database operation fails."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=500,
            context=context,
            **kwargs
        )


class AuthenticationError(DomainException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            status_code=401,
            **kwargs
        )


class AuthorizationError(DomainException):
    """Raised when authorization is denied."""

    def __init__(
        self,
        message: str = "Authorization denied",
        resource: str | None = None,
        action: str | None = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if resource:
            context["resource"] = resource
        if action:
            context["action"] = action

        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_DENIED,
            status_code=403,
            context=context,
            **kwargs
        )

