"""
Authentication Utilities

JWT token generation/validation and password hashing using industry-standard libraries.
"""

from datetime import datetime, timedelta
from uuid import UUID

import bcrypt
import structlog
from jose import JWTError, jwt

from shared.config import settings

logger = structlog.get_logger(__name__)


class PasswordHasher:
    """Service for secure password hashing and verification using bcrypt directly."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        BCrypt has a maximum password length of 72 bytes. We truncate to ensure
        compatibility across all bcrypt implementations.

        Args:
            password: Plain text password

        Returns:
            str: Hashed password
        """
        # Truncate password to 72 bytes for bcrypt compatibility
        password_bytes = password.encode('utf-8')[:72]

        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)

        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            bool: True if password matches
        """
        # Truncate password to 72 bytes for bcrypt compatibility
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')

        return bcrypt.checkpw(password_bytes, hashed_bytes)


class JWTManager:
    """Service for JWT token generation and validation."""

    @staticmethod
    def create_access_token(
        user_id: UUID,
        email: str,
        roles: list[str],
        user_type: str | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create JWT access token with role-based expiration.

        Args:
            user_id: User UUID
            email: User email
            roles: User roles
            user_type: User type (student, lecturer, staff, admin)
            expires_delta: Token expiration time (overrides role-based default)

        Returns:
            str: JWT token
        """
        if expires_delta is None:
            # Role-based expiration times (in minutes)
            role_expiry = {
                "student": 60,  # 1 hour
                "lecturer": 120,  # 2 hours
                "staff": 90,  # 1.5 hours
                "admin": 30,  # 30 minutes (shorter for security)
            }

            if user_type and user_type.lower() in role_expiry:
                expires_delta = timedelta(minutes=role_expiry[user_type.lower()])
            else:
                expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)

        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "sub": str(user_id),
            "email": email,
            "roles": roles,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)

        logger.debug("Access token created", user_id=str(user_id), expires_at=expire.isoformat())

        return encoded_jwt

    @staticmethod
    def create_refresh_token(user_id: UUID) -> str:
        """
        Create JWT refresh token.

        Args:
            user_id: User UUID

        Returns:
            str: Refresh token
        """
        expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)

        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)

        logger.debug("Refresh token created", user_id=str(user_id), expires_at=expire.isoformat())

        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> dict:
        """
        Decode and validate JWT token.

        Args:
            token: JWT token string

        Returns:
            dict: Token payload

        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        except JWTError as e:
            logger.warning("Token validation failed", error=str(e))
            raise TokenError(f"Invalid token: {str(e)}")

    @staticmethod
    def get_user_id_from_token(token: str) -> UUID:
        """
        Extract user ID from token.

        Args:
            token: JWT token

        Returns:
            UUID: User ID

        Raises:
            TokenError: If token is invalid
        """
        payload = JWTManager.decode_token(token)
        user_id_str = payload.get("sub")

        if not user_id_str:
            raise TokenError("Token missing subject (user ID)")

        return UUID(user_id_str)

    @staticmethod
    def verify_token_type(token: str, expected_type: str) -> bool:
        """
        Verify token type (access vs refresh).

        Args:
            token: JWT token
            expected_type: Expected token type

        Returns:
            bool: True if token type matches
        """
        payload = JWTManager.decode_token(token)
        token_type = payload.get("type", "access")
        return token_type == expected_type


class TokenError(Exception):
    """Raised when token validation fails."""



# Convenience instances
password_hasher = PasswordHasher()
jwt_manager = JWTManager()

