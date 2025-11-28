"""
Policy Engine & Enrollment Policies

Implements Strategy pattern for pluggable enrollment policies.
Supports prerequisite checking, quota enforcement, and priority enrollment.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class PolicyResult(BaseModel):
    """Result of policy evaluation."""

    model_config = ConfigDict(frozen=True)

    allowed: bool = Field(..., description="Whether action is allowed")
    reason: str = Field(..., description="Human-readable reason")
    violated_rules: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnrollmentPolicy(ABC):
    """
    Abstract base class for enrollment policies (Strategy pattern).

    Each concrete policy implements specific enrollment rules.
    Policies can be chained for complex validation.
    """

    def __init__(self, name: str, priority: int = 0):
        """
        Initialize policy.

        Args:
            name: Policy identifier
            priority: Execution priority (higher = earlier)
        """
        self.name = name
        self.priority = priority

    @abstractmethod
    async def evaluate(
        self,
        student_id: UUID,
        section_id: UUID,
        context: dict[str, Any],
    ) -> PolicyResult:
        """
        Evaluate if enrollment is allowed.

        Args:
            student_id: Student attempting to enroll
            section_id: Target section
            context: Additional context (current enrollments, student data, etc.)

        Returns:
            PolicyResult: Evaluation result
        """

    def __lt__(self, other: "EnrollmentPolicy") -> bool:
        """Compare policies by priority for sorting."""
        return self.priority > other.priority  # Higher priority first


class PrerequisitePolicy(EnrollmentPolicy):
    """
    Policy that checks course prerequisites.

    Validates that student has completed all required prerequisite courses.
    """

    def __init__(self, priority: int = 100):
        super().__init__("prerequisite_check", priority)

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check if student has completed prerequisites.

        Context should include:
        - course_prerequisites: list of required course codes
        - student_completed_courses: list of completed course codes
        """
        prerequisites: list[str] = context.get("course_prerequisites", [])
        completed_courses: list[str] = context.get("student_completed_courses", [])

        if not prerequisites:
            return PolicyResult(
                allowed=True, reason="No prerequisites required", metadata={}
            )

        missing = [prereq for prereq in prerequisites if prereq not in completed_courses]

        if missing:
            return PolicyResult(
                allowed=False,
                reason=f"Missing prerequisites: {', '.join(missing)}",
                violated_rules=["prerequisite_requirement"],
                metadata={"missing_prerequisites": missing},
            )

        return PolicyResult(
            allowed=True,
            reason="All prerequisites satisfied",
            metadata={"prerequisites_checked": prerequisites},
        )


class CapacityPolicy(EnrollmentPolicy):
    """
    Policy that enforces section capacity limits.

    Checks if section has available seats.
    """

    def __init__(self, priority: int = 90):
        super().__init__("capacity_check", priority)

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check if section has capacity.

        Context should include:
        - section_max_enrollment: int
        - section_current_enrollment: int
        """
        max_enrollment: int = context.get("section_max_enrollment", 0)
        current_enrollment: int = context.get("section_current_enrollment", 0)

        if current_enrollment >= max_enrollment:
            return PolicyResult(
                allowed=False,
                reason=f"Section is full ({current_enrollment}/{max_enrollment})",
                violated_rules=["capacity_limit"],
                metadata={
                    "max_enrollment": max_enrollment,
                    "current_enrollment": current_enrollment,
                },
            )

        return PolicyResult(
            allowed=True,
            reason=f"Capacity available ({current_enrollment}/{max_enrollment})",
            metadata={
                "available_seats": max_enrollment - current_enrollment,
            },
        )


class TimeConflictPolicy(EnrollmentPolicy):
    """
    Policy that prevents schedule conflicts.

    Ensures student doesn't enroll in overlapping sections.
    """

    def __init__(self, priority: int = 95):
        super().__init__("time_conflict_check", priority)

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check for schedule conflicts.

        Context should include:
        - section_schedule: dict with days, start_time, end_time
        - student_current_schedule: list of enrolled section schedules
        """
        section_schedule: dict[str, Any] = context.get("section_schedule", {})
        current_schedule: list[dict[str, Any]] = context.get("student_current_schedule", [])

        section_days = set(section_schedule.get("days", []))
        section_start = section_schedule.get("start_time", "")
        section_end = section_schedule.get("end_time", "")

        for enrolled_section in current_schedule:
            enrolled_days = set(enrolled_section.get("days", []))
            enrolled_start = enrolled_section.get("start_time", "")
            enrolled_end = enrolled_section.get("end_time", "")

            # Check for day overlap
            if not section_days.intersection(enrolled_days):
                continue

            # Check for time overlap
            if self._times_overlap(section_start, section_end, enrolled_start, enrolled_end):
                return PolicyResult(
                    allowed=False,
                    reason=f"Schedule conflict with {enrolled_section.get('course_code', 'another course')}",
                    violated_rules=["no_time_conflict"],
                    metadata={
                        "conflicting_section": enrolled_section.get("section_id"),
                        "conflicting_course": enrolled_section.get("course_code"),
                    },
                )

        return PolicyResult(
            allowed=True, reason="No schedule conflicts detected", metadata={}
        )

    def _times_overlap(self, start1: str, end1: str, start2: str, end2: str) -> bool:
        """Check if two time ranges overlap."""
        # Simple string comparison (assumes HH:MM format)
        return start1 < end2 and start2 < end1


class CreditLimitPolicy(EnrollmentPolicy):
    """
    Policy that enforces credit hour limits per semester.

    Prevents students from overloading their schedule.
    """

    def __init__(self, max_credits: int = 18, priority: int = 80):
        super().__init__("credit_limit_check", priority)
        self.max_credits = max_credits

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check if enrollment exceeds credit limit.

        Context should include:
        - course_credits: int
        - student_current_credits: int
        """
        course_credits: int = context.get("course_credits", 0)
        current_credits: int = context.get("student_current_credits", 0)

        total_credits = current_credits + course_credits

        if total_credits > self.max_credits:
            return PolicyResult(
                allowed=False,
                reason=f"Exceeds credit limit ({total_credits}/{self.max_credits})",
                violated_rules=["credit_limit"],
                metadata={
                    "max_credits": self.max_credits,
                    "current_credits": current_credits,
                    "course_credits": course_credits,
                    "total_credits": total_credits,
                },
            )

        return PolicyResult(
            allowed=True,
            reason=f"Within credit limit ({total_credits}/{self.max_credits})",
            metadata={"total_credits": total_credits},
        )


class AcademicStandingPolicy(EnrollmentPolicy):
    """
    Policy that checks student's academic standing.

    Students on probation may have enrollment restrictions.
    """

    def __init__(self, priority: int = 85):
        super().__init__("academic_standing_check", priority)

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check academic standing requirements.

        Context should include:
        - student_academic_standing: str (good/probation/suspended)
        - student_gpa: float
        """
        standing: str = context.get("student_academic_standing", "good")
        gpa: float = context.get("student_gpa", 0.0)

        if standing == "suspended":
            return PolicyResult(
                allowed=False,
                reason="Student is academically suspended",
                violated_rules=["academic_standing"],
                metadata={"standing": standing, "gpa": gpa},
            )

        if standing == "probation":
            # Students on probation have reduced credit limits
            return PolicyResult(
                allowed=True,
                reason="Student on academic probation - enrollment allowed with restrictions",
                metadata={
                    "standing": standing,
                    "gpa": gpa,
                    "warning": "Academic probation - limited credit hours",
                },
            )

        return PolicyResult(
            allowed=True, reason="Student in good academic standing", metadata={}
        )


class PriorityEnrollmentPolicy(EnrollmentPolicy):
    """
    Policy that implements priority enrollment.

    Gives enrollment priority based on student attributes
    (seniors first, honors students, athletes, etc.).
    """

    def __init__(self, enrollment_start_date: datetime, priority: int = 70):
        super().__init__("priority_enrollment", priority)
        self.enrollment_start_date = enrollment_start_date

    async def evaluate(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> PolicyResult:
        """
        Check if student can enroll based on priority.

        Context should include:
        - student_priority_group: str (senior, junior, sophomore, freshman)
        - student_is_honors: bool
        - current_time: datetime
        """
        priority_group: str = context.get("student_priority_group", "freshman")
        is_honors: bool = context.get("student_is_honors", False)
        current_time: datetime = context.get("current_time", datetime.utcnow())

        # Priority enrollment windows (days before general enrollment)
        priority_windows = {
            "senior": 7,
            "junior": 5,
            "sophomore": 3,
            "freshman": 0,
        }

        if is_honors:
            priority_windows[priority_group] += 2  # Honors get +2 days

        window_days = priority_windows.get(priority_group, 0)
        enrollment_opens_at = self.enrollment_start_date

        if current_time < enrollment_opens_at:
            days_early = (enrollment_opens_at - current_time).days
            if days_early > window_days:
                return PolicyResult(
                    allowed=False,
                    reason=f"Enrollment opens in {days_early} days. "
                    f"Your priority window is {window_days} days.",
                    violated_rules=["priority_enrollment_window"],
                    metadata={
                        "enrollment_opens_at": enrollment_opens_at.isoformat(),
                        "priority_group": priority_group,
                        "priority_window_days": window_days,
                    },
                )

        return PolicyResult(
            allowed=True,
            reason=f"Priority enrollment window open for {priority_group}",
            metadata={"priority_group": priority_group},
        )


class PolicyEngine:
    """
    Policy evaluation engine that coordinates multiple policies.

    Executes policies in priority order and aggregates results.
    """

    def __init__(self):
        """Initialize policy engine."""
        self.policies: list[EnrollmentPolicy] = []

    def register_policy(self, policy: EnrollmentPolicy) -> None:
        """
        Register a policy with the engine.

        Args:
            policy: Policy to register
        """
        self.policies.append(policy)
        self.policies.sort()  # Sort by priority
        logger.info("Policy registered", policy_name=policy.name, priority=policy.priority)

    def unregister_policy(self, policy_name: str) -> bool:
        """
        Unregister a policy.

        Args:
            policy_name: Name of policy to remove

        Returns:
            bool: True if policy was found and removed
        """
        initial_count = len(self.policies)
        self.policies = [p for p in self.policies if p.name != policy_name]
        return len(self.policies) < initial_count

    async def evaluate_all(
        self, student_id: UUID, section_id: UUID, context: dict[str, Any]
    ) -> tuple[bool, list[PolicyResult]]:
        """
        Evaluate all registered policies.

        Args:
            student_id: Student ID
            section_id: Section ID
            context: Evaluation context

        Returns:
            Tuple of (all_allowed, list of results)
        """
        results: list[PolicyResult] = []

        for policy in self.policies:
            try:
                result = await policy.evaluate(student_id, section_id, context)
                results.append(result)

                # Stop on first failure (fail-fast)
                if not result.allowed:
                    logger.info(
                        "Policy evaluation failed",
                        policy=policy.name,
                        student_id=str(student_id),
                        section_id=str(section_id),
                        reason=result.reason,
                    )
                    return False, results

            except Exception as e:
                logger.error(
                    "Policy evaluation error",
                    policy=policy.name,
                    error=str(e),
                    student_id=str(student_id),
                    section_id=str(section_id),
                )
                # On error, create denied result
                error_result = PolicyResult(
                    allowed=False,
                    reason=f"Policy evaluation error: {str(e)}",
                    violated_rules=["policy_execution_error"],
                    metadata={"policy": policy.name, "error": str(e)},
                )
                results.append(error_result)
                return False, results

        # All policies passed
        all_allowed = all(r.allowed for r in results)
        return all_allowed, results

    def get_registered_policies(self) -> list[str]:
        """Get list of registered policy names."""
        return [p.name for p in self.policies]


# Default enrollment policy engine with common policies
def create_default_enrollment_policy_engine() -> PolicyEngine:
    """
    Create policy engine with default enrollment policies.

    Returns:
        PolicyEngine: Configured engine with standard policies
    """
    engine = PolicyEngine()

    # Register policies in priority order (higher priority = execute first)
    engine.register_policy(PrerequisitePolicy(priority=100))
    engine.register_policy(TimeConflictPolicy(priority=95))
    engine.register_policy(CapacityPolicy(priority=90))
    engine.register_policy(AcademicStandingPolicy(priority=85))
    engine.register_policy(CreditLimitPolicy(priority=80))

    logger.info(
        "Default enrollment policy engine created",
        policies=engine.get_registered_policies(),
    )

    return engine

