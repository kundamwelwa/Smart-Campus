"""
Scheduler Subsystem with Constraints and Timetable Snapshotting

Implements constraint-based scheduling with soft/hard constraints
and timetable snapshotting for version control.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from shared.domain.entities import VersionedEntity


class ConstraintType(str, Enum):
    """Type of scheduling constraint."""

    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preferred but not required


class ConstraintPriority(int, Enum):
    """Constraint priority for soft constraints."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Constraint(BaseModel):
    """
    Scheduling constraint for timetable generation.

    Constraints can be hard (must be satisfied) or soft (preferred).
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4, description="Constraint unique ID")
    name: str = Field(..., description="Constraint name/identifier")
    constraint_type: ConstraintType = Field(..., description="Hard or soft constraint")
    priority: ConstraintPriority = Field(
        default=ConstraintPriority.MEDIUM, description="Priority for soft constraints"
    )
    weight: float = Field(
        default=1.0, ge=0.0, description="Weight for optimization (higher = more important)"
    )
    description: str = Field(..., max_length=500)
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Constraint-specific parameters"
    )

    def is_hard(self) -> bool:
        """Check if this is a hard constraint."""
        return self.constraint_type == ConstraintType.HARD

    def is_soft(self) -> bool:
        """Check if this is a soft constraint."""
        return self.constraint_type == ConstraintType.SOFT


class CapacityConstraint(Constraint):
    """Hard constraint: Room capacity must accommodate section size."""

    def __init__(self, **data: Any):
        super().__init__(
            name="capacity_constraint",
            constraint_type=ConstraintType.HARD,
            description="Room capacity must be sufficient for section enrollment",
            **data,
        )


class TimeConflictConstraint(Constraint):
    """Hard constraint: No overlapping time slots for same room/instructor."""

    def __init__(self, **data: Any):
        super().__init__(
            name="time_conflict_constraint",
            constraint_type=ConstraintType.HARD,
            description="No overlapping time slots for same resource",
            **data,
        )


class InstructorAvailabilityConstraint(Constraint):
    """Hard constraint: Instructor must be available at scheduled time."""

    def __init__(self, **data: Any):
        super().__init__(
            name="instructor_availability_constraint",
            constraint_type=ConstraintType.HARD,
            description="Instructor must be available",
            **data,
        )


class RoomPreferenceConstraint(Constraint):
    """Soft constraint: Prefer certain rooms for certain courses."""

    def __init__(self, **data: Any):
        super().__init__(
            name="room_preference_constraint",
            constraint_type=ConstraintType.SOFT,
            priority=ConstraintPriority.MEDIUM,
            weight=0.5,
            description="Prefer specific rooms for courses",
            **data,
        )


class BalancedWorkloadConstraint(Constraint):
    """Soft constraint: Balance instructor workload across time slots."""

    def __init__(self, **data: Any):
        super().__init__(
            name="balanced_workload_constraint",
            constraint_type=ConstraintType.SOFT,
            priority=ConstraintPriority.LOW,
            weight=0.3,
            description="Balance instructor workload",
            **data,
        )


class TimetableSnapshot(VersionedEntity):
    """
    Immutable snapshot of a timetable at a point in time.

    Enables version control, rollback, and historical analysis.
    """

    semester: str = Field(..., description="Semester (e.g., Fall 2024)")
    snapshot_date: date = Field(default_factory=date.today)
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)

    # Timetable Data
    assignments: dict[str, Any] = Field(
        ..., description="Section-to-room-time assignments"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional snapshot metadata"
    )

    # Optimization Results
    optimization_score: float = Field(
        default=0.0, description="Overall optimization score"
    )
    hard_constraints_satisfied: bool = Field(
        default=True, description="Whether all hard constraints are satisfied"
    )
    soft_constraints_score: float = Field(
        default=0.0, description="Score for soft constraint satisfaction"
    )
    constraint_violations: list[str] = Field(
        default_factory=list, description="List of violated constraints"
    )

    # Relationships
    created_by: UUID | None = Field(default=None, description="User who created snapshot")
    parent_snapshot_id: UUID | None = Field(
        default=None, description="Parent snapshot ID (for versioning)"
    )

    def validate_business_rules(self) -> bool:
        """Validate timetable snapshot business rules."""
        if self.optimization_score < 0.0 or self.optimization_score > 1.0:
            raise ValueError("Optimization score must be between 0.0 and 1.0")
        return True

    def get_assignment_count(self) -> int:
        """Get number of assignments in this snapshot."""
        return len(self.assignments)

    def is_valid(self) -> bool:
        """Check if timetable is valid (all hard constraints satisfied)."""
        return self.hard_constraints_satisfied and len(self.constraint_violations) == 0


class Timetable(VersionedEntity):
    """
    Current timetable with constraint management and snapshotting.

    Supports:
    - Constraint-based scheduling
    - Timetable snapshots for version control
    - Rollback to previous snapshots
    """

    semester: str = Field(..., description="Semester identifier")
    academic_year: str = Field(..., description="Academic year (e.g., 2024-2025)")

    # Current State
    assignments: dict[str, Any] = Field(
        default_factory=dict, description="Current section assignments"
    )
    constraints: list[Constraint] = Field(
        default_factory=list, description="Active constraints"
    )

    # Snapshot Management
    snapshots: list[UUID] = Field(
        default_factory=list, description="Snapshot IDs (version history)"
    )
    current_snapshot_id: UUID | None = Field(
        default=None, description="Current snapshot ID"
    )

    # Status
    is_finalized: bool = Field(default=False, description="Whether timetable is finalized")
    finalized_at: datetime | None = Field(default=None)
    finalized_by: UUID | None = Field(default=None)

    def validate_business_rules(self) -> bool:
        """Validate timetable business rules."""
        if self.is_finalized and not self.finalized_at:
            raise ValueError("Finalized timetable must have finalized_at timestamp")
        return True

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the timetable."""
        if constraint.id not in [c.id for c in self.constraints]:
            self.constraints.append(constraint)
            self.mark_updated()

    def remove_constraint(self, constraint_id: UUID) -> bool:
        """
        Remove a constraint.

        Args:
            constraint_id: Constraint ID to remove

        Returns:
            True if constraint was found and removed
        """
        initial_count = len(self.constraints)
        self.constraints = [c for c in self.constraints if c.id != constraint_id]
        removed = len(self.constraints) < initial_count
        if removed:
            self.mark_updated()
        return removed

    def get_hard_constraints(self) -> list[Constraint]:
        """Get all hard constraints."""
        return [c for c in self.constraints if c.is_hard()]

    def get_soft_constraints(self) -> list[Constraint]:
        """Get all soft constraints."""
        return [c for c in self.constraints if c.is_soft()]

    def create_snapshot(
        self,
        optimization_score: float = 0.0,
        hard_constraints_satisfied: bool = True,
        soft_constraints_score: float = 0.0,
        constraint_violations: list[str] | None = None,
        created_by: UUID | None = None,
    ) -> TimetableSnapshot:
        """
        Create a snapshot of the current timetable state.

        Args:
            optimization_score: Overall optimization score
            hard_constraints_satisfied: Whether hard constraints are satisfied
            soft_constraints_score: Soft constraint satisfaction score
            constraint_violations: List of violated constraints
            created_by: User creating the snapshot

        Returns:
            TimetableSnapshot: Created snapshot
        """
        snapshot = TimetableSnapshot(
            semester=self.semester,
            assignments=self.assignments.copy(),
            metadata={
                "academic_year": self.academic_year,
                "constraint_count": len(self.constraints),
                "assignment_count": len(self.assignments),
            },
            optimization_score=optimization_score,
            hard_constraints_satisfied=hard_constraints_satisfied,
            soft_constraints_score=soft_constraints_score,
            constraint_violations=constraint_violations or [],
            parent_snapshot_id=self.current_snapshot_id,
            created_by=created_by,
        )

        # Add to snapshot history
        self.snapshots.append(snapshot.id)
        self.current_snapshot_id = snapshot.id
        self.mark_updated()

        return snapshot

    def finalize(self, finalized_by: UUID) -> None:
        """
        Finalize the timetable (make it immutable).

        Args:
            finalized_by: User finalizing the timetable
        """
        if self.is_finalized:
            raise ValueError("Timetable is already finalized")

        self.is_finalized = True
        self.finalized_at = datetime.utcnow()
        self.finalized_by = finalized_by
        self.mark_updated()

    def can_modify(self) -> bool:
        """Check if timetable can be modified."""
        return not self.is_finalized

