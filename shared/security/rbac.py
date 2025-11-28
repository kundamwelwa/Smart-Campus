"""
Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC)

Fine-grained authorization services supporting both role-based and attribute-based access control.
"""

from typing import Any
from uuid import UUID

import structlog

from shared.domain.security import Permission, PermissionAction, ResourceType, Role

logger = structlog.get_logger(__name__)


class RBACService:
    """
    Role-Based Access Control service.

    Evaluates permissions based on user roles and role hierarchies.
    """

    def __init__(self):
        """Initialize RBAC service."""
        self.role_cache: dict[UUID, Role] = {}

    def load_role(self, role: Role) -> None:
        """
        Load a role into the cache.

        Args:
            role: Role to cache
        """
        self.role_cache[role.id] = role

    def has_permission(
        self,
        user_roles: list[UUID],
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: UUID | None = None,
    ) -> bool:
        """
        Check if user has permission based on their roles.

        Args:
            user_roles: List of role IDs assigned to user
            action: Required permission action
            resource_type: Resource type to access
            resource_id: Specific resource ID (optional)

        Returns:
            bool: True if user has permission
        """
        # Check each role
        for role_id in user_roles:
            role = self.role_cache.get(role_id)
            if role is None:
                logger.warning("Role not found in cache", role_id=str(role_id))
                continue

            # Check direct permissions
            if role.has_permission(action, resource_type, resource_id):
                logger.debug(
                    "Permission granted via role",
                    role_name=role.name,
                    action=action.value,
                    resource_type=resource_type.value,
                )
                return True

            # Check inherited roles (recursive)
            if self._check_inherited_permissions(
                role, action, resource_type, resource_id
            ):
                return True

        logger.debug(
            "Permission denied",
            action=action.value,
            resource_type=resource_type.value,
            user_roles_count=len(user_roles),
        )
        return False

    def _check_inherited_permissions(
        self,
        role: Role,
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: UUID | None = None,
    ) -> bool:
        """Recursively check permissions from inherited roles."""
        for parent_role_id in role.inherits_from:
            parent_role = self.role_cache.get(parent_role_id)
            if parent_role is None:
                continue

            if parent_role.has_permission(action, resource_type, resource_id):
                return True

            # Recursive check
            if self._check_inherited_permissions(
                parent_role, action, resource_type, resource_id
            ):
                return True

        return False

    def get_effective_permissions(self, user_roles: list[UUID]) -> list[Permission]:
        """
        Get all effective permissions for a user (including inherited).

        Args:
            user_roles: List of role IDs

        Returns:
            List of permissions
        """
        all_permissions: list[Permission] = []

        for role_id in user_roles:
            role = self.role_cache.get(role_id)
            if role:
                all_permissions.extend(role.permissions)
                # Add inherited permissions
                all_permissions.extend(self._get_inherited_permissions(role))

        # Remove duplicates
        return list({str(p): p for p in all_permissions}.values())

    def _get_inherited_permissions(self, role: Role) -> list[Permission]:
        """Recursively collect inherited permissions."""
        inherited: list[Permission] = []

        for parent_role_id in role.inherits_from:
            parent_role = self.role_cache.get(parent_role_id)
            if parent_role:
                inherited.extend(parent_role.permissions)
                inherited.extend(self._get_inherited_permissions(parent_role))

        return inherited


class ABACService:
    """
    Attribute-Based Access Control service.

    Evaluates permissions based on attributes of:
    - Subject (user attributes)
    - Resource (resource attributes)
    - Action (action attributes)
    - Environment (time, location, etc.)
    """

    def __init__(self):
        """Initialize ABAC service."""
        self.policies: list[dict[str, Any]] = []

    def add_policy(self, policy: dict[str, Any]) -> None:
        """
        Add an ABAC policy.

        Policy format:
        {
            "name": "policy_name",
            "rules": [
                {
                    "subject": {"department": "CS"},
                    "resource": {"type": "course", "department": "CS"},
                    "action": "read",
                    "effect": "allow"
                }
            ]
        }

        Args:
            policy: Policy definition
        """
        self.policies.append(policy)

    def evaluate(
        self,
        subject_attributes: dict[str, Any],
        resource_attributes: dict[str, Any],
        action: str,
        environment_attributes: dict[str, Any] | None = None,
    ) -> bool:
        """
        Evaluate ABAC policies.

        Args:
            subject_attributes: User/subject attributes
            resource_attributes: Resource attributes
            action: Action being performed
            environment_attributes: Environmental attributes (time, location, etc.)

        Returns:
            bool: True if access is allowed
        """
        env_attrs = environment_attributes or {}

        for policy in self.policies:
            for rule in policy.get("rules", []):
                if self._rule_matches(
                    rule, subject_attributes, resource_attributes, action, env_attrs
                ):
                    effect = rule.get("effect", "deny")
                    if effect == "allow":
                        logger.debug(
                            "ABAC policy matched - access allowed",
                            policy_name=policy.get("name"),
                        )
                        return True

        logger.debug("No matching ABAC policy - access denied")
        return False

    def _rule_matches(
        self,
        rule: dict[str, Any],
        subject_attrs: dict[str, Any],
        resource_attrs: dict[str, Any],
        action: str,
        env_attrs: dict[str, Any],
    ) -> bool:
        """Check if a rule matches the request."""
        # Check action
        if rule.get("action") != action:
            return False

        # Check subject attributes
        if not self._attributes_match(rule.get("subject", {}), subject_attrs):
            return False

        # Check resource attributes
        if not self._attributes_match(rule.get("resource", {}), resource_attrs):
            return False

        # Check environment attributes (if specified in rule)
        return self._attributes_match(rule.get("environment", {}), env_attrs)

    def _attributes_match(
        self, required_attrs: dict[str, Any], actual_attrs: dict[str, Any]
    ) -> bool:
        """Check if actual attributes satisfy required attributes."""
        for key, required_value in required_attrs.items():
            actual_value = actual_attrs.get(key)

            # Handle list containment
            if isinstance(required_value, list):
                if actual_value not in required_value:
                    return False
            # Handle exact match
            elif actual_value != required_value:
                return False

        return True


class AuthorizationService:
    """
    Combined authorization service using both RBAC and ABAC.

    Evaluates permissions using both role-based and attribute-based policies,
    providing comprehensive fine-grained access control.
    """

    def __init__(self, rbac: RBACService, abac: ABACService):
        """
        Initialize authorization service.

        Args:
            rbac: RBAC service instance
            abac: ABAC service instance
        """
        self.rbac = rbac
        self.abac = abac

    async def authorize(
        self,
        user_id: UUID,
        user_roles: list[UUID],
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: UUID | None = None,
        subject_attributes: dict[str, Any] | None = None,
        resource_attributes: dict[str, Any] | None = None,
        environment_attributes: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Authorize an action using combined RBAC and ABAC.

        First checks RBAC (faster), then falls back to ABAC if needed.

        Args:
            user_id: User requesting access
            user_roles: User's role IDs
            action: Requested action
            resource_type: Target resource type
            resource_id: Specific resource ID
            subject_attributes: Additional user attributes for ABAC
            resource_attributes: Resource attributes for ABAC
            environment_attributes: Environment attributes for ABAC

        Returns:
            Tuple of (is_authorized, reason)
        """
        # Try RBAC first (faster, more common)
        if self.rbac.has_permission(user_roles, action, resource_type, resource_id):
            logger.info(
                "Access authorized via RBAC",
                user_id=str(user_id),
                action=action.value,
                resource_type=resource_type.value,
            )
            return True, "Authorized by role-based policy"

        # Fall back to ABAC for fine-grained control
        if subject_attributes and resource_attributes and self.abac.evaluate(
            subject_attributes=subject_attributes or {},
            resource_attributes=resource_attributes or {},
            action=action.value,
            environment_attributes=environment_attributes,
        ):
            logger.info(
                "Access authorized via ABAC",
                user_id=str(user_id),
                action=action.value,
                resource_type=resource_type.value,
            )
            return True, "Authorized by attribute-based policy"

        # Access denied
        logger.warning(
            "Access denied",
            user_id=str(user_id),
            action=action.value,
            resource_type=resource_type.value,
        )
        return False, "Insufficient permissions"


# Create default roles
def create_default_roles() -> dict[str, Role]:
    """
    Create default system roles.

    Returns:
        dict: Dictionary of role name to Role object
    """
    from shared.domain.entities import uuid4

    # Student Role
    student_role = Role(
        id=uuid4(),
        name="student",
        description="Default student role with basic access",
        permissions=[
            Permission(action=PermissionAction.READ, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.CREATE, resource_type=ResourceType.ENROLLMENT),
            Permission(action=PermissionAction.READ, resource_type=ResourceType.GRADE),
            Permission(action=PermissionAction.CREATE, resource_type=ResourceType.BOOKING),
        ],
        is_system_role=True,
        priority=10,
    )

    # Lecturer Role
    lecturer_role = Role(
        id=uuid4(),
        name="lecturer",
        description="Lecturer with teaching permissions",
        permissions=[
            Permission(action=PermissionAction.READ, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.UPDATE, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.CREATE, resource_type=ResourceType.GRADE),
            Permission(action=PermissionAction.UPDATE, resource_type=ResourceType.GRADE),
            Permission(action=PermissionAction.READ, resource_type=ResourceType.STUDENT),
            Permission(action=PermissionAction.READ, resource_type=ResourceType.ENROLLMENT),
        ],
        is_system_role=True,
        priority=50,
    )

    # Admin Role
    admin_role = Role(
        id=uuid4(),
        name="admin",
        description="System administrator with full access",
        permissions=[
            Permission(action=PermissionAction.CREATE, resource_type=ResourceType.USER),
            Permission(action=PermissionAction.READ, resource_type=ResourceType.USER),
            Permission(action=PermissionAction.UPDATE, resource_type=ResourceType.USER),
            Permission(action=PermissionAction.DELETE, resource_type=ResourceType.USER),
            Permission(action=PermissionAction.CREATE, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.UPDATE, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.DELETE, resource_type=ResourceType.COURSE),
            Permission(action=PermissionAction.READ, resource_type=ResourceType.AUDIT_LOG),
            Permission(action=PermissionAction.APPROVE, resource_type=ResourceType.ENROLLMENT),
        ],
        is_system_role=True,
        priority=100,
    )

    return {
        "student": student_role,
        "lecturer": lecturer_role,
        "admin": admin_role,
    }

