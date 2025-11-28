"""
Formal Verification Module

Provides runtime verification of critical system invariants.
"""

from shared.verification.enrollment_invariants import (
    Enrollment,
    InvariantMonitor,
    InvariantViolationType,
    Section,
    TimeSlot,
    assert_enrollment_invariant,
    get_invariant_monitor,
)

__all__ = [
    'InvariantMonitor',
    'get_invariant_monitor',
    'assert_enrollment_invariant',
    'Section',
    'TimeSlot',
    'Enrollment',
    'InvariantViolationType',
]

