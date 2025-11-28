"""
End-to-End Encryption for Sensitive Fields

Implements field-level encryption for sensitive data like grades, personal information,
and credentials. Uses Fernet (symmetric) encryption for fields and public-key encryption
for sensitive communications.
"""

import base64
import json
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Try to import cryptography
CRYPTO_AVAILABLE = False
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("Cryptography library not available - encryption disabled")


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive fields.

    Uses symmetric encryption (Fernet) for field-level encryption with
    key rotation support.
    """

    def __init__(self, master_key: bytes | None = None):
        """
        Initialize encryption service.

        Args:
            master_key: Master encryption key (32 bytes). If None, generates new key.
        """
        if not CRYPTO_AVAILABLE:
            logger.error("Encryption not available - cryptography not installed")
            self.cipher = None
            return

        if master_key is None:
            master_key = Fernet.generate_key()

        self.cipher = Fernet(master_key)
        self.master_key = master_key
        logger.info("Encryption service initialized")

    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a new encryption key.

        Returns:
            bytes: 32-byte encryption key
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography not available")
        return Fernet.generate_key()

    def encrypt_field(self, value: Any) -> str:
        """
        Encrypt a field value.

        Args:
            value: Value to encrypt (will be JSON-serialized)

        Returns:
            str: Base64-encoded encrypted value

        Raises:
            RuntimeError: If encryption not available
        """
        if not CRYPTO_AVAILABLE or not self.cipher:
            raise RuntimeError("Encryption not available")

        # Convert value to JSON string
        json_str = json.dumps(value, default=str)
        json_bytes = json_str.encode("utf-8")

        # Encrypt
        encrypted_bytes = self.cipher.encrypt(json_bytes)

        # Return base64-encoded string for storage
        return base64.b64encode(encrypted_bytes).decode("utf-8")

    def decrypt_field(self, encrypted_value: str) -> Any:
        """
        Decrypt a field value.

        Args:
            encrypted_value: Base64-encoded encrypted value

        Returns:
            Decrypted value

        Raises:
            RuntimeError: If encryption not available
            ValueError: If decryption fails
        """
        if not CRYPTO_AVAILABLE or not self.cipher:
            raise RuntimeError("Encryption not available")

        try:
            # Decode base64
            encrypted_bytes = base64.b64decode(encrypted_value.encode("utf-8"))

            # Decrypt
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)

            # Parse JSON
            json_str = decrypted_bytes.decode("utf-8")
            return json.loads(json_str)

        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise ValueError(f"Decryption failed: {str(e)}")

    def encrypt_grade(self, points_earned: float, total_points: float, feedback: str | None = None) -> dict[str, str]:
        """
        Encrypt grade data (example use case).

        Args:
            points_earned: Points earned
            total_points: Total possible points
            feedback: Optional feedback text

        Returns:
            Dictionary with encrypted fields
        """
        grade_data = {
            "points_earned": points_earned,
            "total_points": total_points,
            "feedback": feedback,
        }

        return {
            "encrypted_grade": self.encrypt_field(grade_data),
            "encryption_version": "1",
        }

    def decrypt_grade(self, encrypted_grade: str) -> dict[str, Any]:
        """
        Decrypt grade data.

        Args:
            encrypted_grade: Encrypted grade string

        Returns:
            Dictionary with decrypted grade data
        """
        return self.decrypt_field(encrypted_grade)

    def rotate_key(self, new_key: bytes) -> None:
        """
        Rotate encryption key.

        Note: Existing encrypted data will need to be re-encrypted with new key.

        Args:
            new_key: New encryption key
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Encryption not available")

        self.cipher = Fernet(new_key)
        self.master_key = new_key
        logger.info("Encryption key rotated")


class AsymmetricEncryptionService:
    """
    Service for asymmetric (public/private key) encryption.

    Used for encrypting sensitive communications and key exchange.
    """

    def __init__(self, private_key: bytes | None = None, public_key: bytes | None = None):
        """
        Initialize asymmetric encryption service.

        Args:
            private_key: PEM-encoded private key
            public_key: PEM-encoded public key
        """
        if not CRYPTO_AVAILABLE:
            logger.error("Encryption not available - cryptography not installed")
            self.private_key = None
            self.public_key = None
            return

        if private_key and public_key:
            self.private_key = serialization.load_pem_private_key(
                private_key, password=None, backend=default_backend()
            )
            self.public_key = serialization.load_pem_public_key(
                public_key, backend=default_backend()
            )
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            self.public_key = self.private_key.public_key()

        logger.info("Asymmetric encryption service initialized")

    @classmethod
    def generate_key_pair(cls) -> tuple[bytes, bytes]:
        """
        Generate a new RSA key pair.

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography not available")

        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    def encrypt_with_public_key(self, data: bytes) -> bytes:
        """
        Encrypt data with public key.

        Args:
            data: Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        if not CRYPTO_AVAILABLE or not self.public_key:
            raise RuntimeError("Encryption not available")

        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt_with_private_key(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data with private key.

        Args:
            encrypted_data: Encrypted data

        Returns:
            bytes: Decrypted data
        """
        if not CRYPTO_AVAILABLE or not self.private_key:
            raise RuntimeError("Encryption not available")

        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )


class EncryptedGradeManager:
    """
    Manager for encrypted grade storage and retrieval.

    Handles encryption/decryption of grade data with audit logging.
    """

    def __init__(self, encryption_service: EncryptionService):
        """
        Initialize encrypted grade manager.

        Args:
            encryption_service: Encryption service instance
        """
        self.encryption_service = encryption_service
        self.encrypted_grades: dict[UUID, dict[str, Any]] = {}

    async def store_grade(
        self,
        grade_id: UUID,
        student_id: UUID,
        assessment_id: UUID,
        points_earned: float,
        total_points: float,
        feedback: str | None,
        graded_by: UUID,
    ) -> None:
        """
        Store an encrypted grade.

        Args:
            grade_id: Grade ID
            student_id: Student ID
            assessment_id: Assessment ID
            points_earned: Points earned
            total_points: Total possible points
            feedback: Grade feedback
            graded_by: Grader user ID
        """
        # Encrypt sensitive grade data
        encrypted_data = self.encryption_service.encrypt_grade(
            points_earned, total_points, feedback
        )

        # Store with unencrypted metadata
        self.encrypted_grades[grade_id] = {
            "grade_id": grade_id,
            "student_id": student_id,
            "assessment_id": assessment_id,
            "encrypted_data": encrypted_data["encrypted_grade"],
            "encryption_version": encrypted_data["encryption_version"],
            "graded_by": graded_by,
        }

        logger.info(
            "Encrypted grade stored",
            grade_id=str(grade_id),
            student_id=str(student_id),
        )

    async def retrieve_grade(
        self, grade_id: UUID, requesting_user_id: UUID
    ) -> dict[str, Any] | None:
        """
        Retrieve and decrypt a grade.

        Args:
            grade_id: Grade ID
            requesting_user_id: User requesting the grade (for audit)

        Returns:
            Decrypted grade data or None
        """
        encrypted_grade = self.encrypted_grades.get(grade_id)
        if not encrypted_grade:
            return None

        # Decrypt grade data
        decrypted_data = self.encryption_service.decrypt_grade(
            encrypted_grade["encrypted_data"]
        )

        logger.info(
            "Grade accessed",
            grade_id=str(grade_id),
            accessed_by=str(requesting_user_id),
        )

        return {
            "grade_id": grade_id,
            "student_id": encrypted_grade["student_id"],
            "assessment_id": encrypted_grade["assessment_id"],
            **decrypted_data,
            "graded_by": encrypted_grade["graded_by"],
        }

    async def re_encrypt_all_grades(self, new_encryption_service: EncryptionService) -> int:
        """
        Re-encrypt all grades with new key (for key rotation).

        Args:
            new_encryption_service: New encryption service with rotated key

        Returns:
            int: Number of grades re-encrypted
        """
        count = 0

        for _grade_id, encrypted_grade in self.encrypted_grades.items():
            # Decrypt with old key
            decrypted_data = self.encryption_service.decrypt_grade(
                encrypted_grade["encrypted_data"]
            )

            # Re-encrypt with new key
            re_encrypted_data = new_encryption_service.encrypt_grade(
                decrypted_data["points_earned"],
                decrypted_data["total_points"],
                decrypted_data.get("feedback"),
            )

            # Update stored data
            encrypted_grade["encrypted_data"] = re_encrypted_data["encrypted_grade"]
            encrypted_grade["encryption_version"] = re_encrypted_data[
                "encryption_version"
            ]

            count += 1

        self.encryption_service = new_encryption_service
        logger.info("Grades re-encrypted", count=count)
        return count
