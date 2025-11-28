"""
API Versioning and Backward Compatibility

Implements API versioning strategy with backward compatibility support.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class APIVersion(str, Enum):
    """API version enumeration."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

    @classmethod
    def from_string(cls, version_str: str) -> Optional["APIVersion"]:
        """Parse version string."""
        version_str = version_str.lower().lstrip("v")
        for version in cls:
            if version.value.lstrip("v") == version_str:
                return version
        return None


class VersionedEndpoint:
    """
    Versioned API endpoint with backward compatibility.

    Supports multiple API versions with automatic request/response transformation.
    """

    def __init__(
        self,
        endpoint_name: str,
        current_version: APIVersion = APIVersion.V1,
        deprecated_versions: list[APIVersion] | None = None,
    ):
        """
        Initialize versioned endpoint.

        Args:
            endpoint_name: Endpoint identifier
            current_version: Current API version
            deprecated_versions: List of deprecated versions
        """
        self.endpoint_name = endpoint_name
        self.current_version = current_version
        self.deprecated_versions = deprecated_versions or []
        self.version_handlers: dict[APIVersion, Callable] = {}
        self.transformers: dict[tuple[APIVersion, APIVersion], Callable] = {}

    def register_version(
        self, version: APIVersion, handler: Callable, is_deprecated: bool = False
    ) -> None:
        """
        Register a version handler.

        Args:
            version: API version
            handler: Request handler function
            is_deprecated: Whether this version is deprecated
        """
        self.version_handlers[version] = handler
        if is_deprecated and version not in self.deprecated_versions:
            self.deprecated_versions.append(version)

    def register_transformer(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
        transformer: Callable[[Any], Any],
    ) -> None:
        """
        Register a request/response transformer between versions.

        Args:
            from_version: Source version
            to_version: Target version
            transformer: Transformation function
        """
        self.transformers[(from_version, to_version)] = transformer

    async def handle_request(
        self, request: Any, requested_version: APIVersion | None = None
    ) -> Any:
        """
        Handle request with version routing and transformation.

        Args:
            request: Request object
            requested_version: Requested API version (None = current)

        Returns:
            Response object
        """
        # Default to current version
        version = requested_version or self.current_version

        # Check if version is deprecated
        if version in self.deprecated_versions:
            logger.warning(
                "Deprecated API version used",
                endpoint=self.endpoint_name,
                version=version.value,
            )

        # Get handler for requested version or use transformer
        if version in self.version_handlers:
            handler = self.version_handlers[version]
        elif version != self.current_version:
            # Transform request to current version
            handler = self.version_handlers[self.current_version]
            if (version, self.current_version) in self.transformers:
                request = self.transformers[(version, self.current_version)](request)
        else:
            handler = self.version_handlers[self.current_version]

        # Execute handler
        response = await handler(request)

        # Transform response back to requested version if needed
        if version != self.current_version and (self.current_version, version) in self.transformers:
            response = self.transformers[(self.current_version, version)](response)

        return response


class APIVersionMiddleware:
    """
    Middleware for API version detection and routing.

    Extracts version from headers, query params, or URL path.
    """

    @staticmethod
    def extract_version(
        headers: dict[str, str],
        query_params: dict[str, str],
        path: str,
    ) -> APIVersion | None:
        """
        Extract API version from request.

        Checks in order:
        1. Accept header: application/vnd.argos.v2+json
        2. Query parameter: ?api_version=v2
        3. URL path: /api/v2/endpoint

        Args:
            headers: Request headers
            query_params: Query parameters
            path: URL path

        Returns:
            APIVersion or None
        """
        # Check Accept header
        accept = headers.get("accept", "")
        if "vnd.argos." in accept:
            for version in APIVersion:
                if f"vnd.argos.{version.value}" in accept:
                    return version

        # Check query parameter
        api_version = query_params.get("api_version") or query_params.get("version")
        if api_version:
            return APIVersion.from_string(api_version)

        # Check URL path
        if "/v1/" in path:
            return APIVersion.V1
        if "/v2/" in path:
            return APIVersion.V2
        if "/v3/" in path:
            return APIVersion.V3

        return None


def create_versioned_router(base_path: str = "/api") -> dict[str, str]:
    """
    Create versioned route paths.

    Args:
        base_path: Base API path

    Returns:
        Dictionary mapping versions to paths
    """
    return {
        APIVersion.V1.value: f"{base_path}/v1",
        APIVersion.V2.value: f"{base_path}/v2",
        APIVersion.V3.value: f"{base_path}/v3",
    }

