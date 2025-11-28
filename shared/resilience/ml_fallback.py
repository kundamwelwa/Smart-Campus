"""
Rule-based Fallback Logic for ML Service

Provides rule-based predictions when ML service is unavailable.
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def rule_based_enrollment_prediction(student_data: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based enrollment dropout prediction.

    Falls back to this when ML service is unavailable.

    Args:
        student_data: Student data dictionary with:
            - gpa: GPA (0-4.0)
            - credits_enrolled: Number of credits enrolled
            - attendance_rate: Attendance rate (0-100)
            - engagement_score: Engagement score (0-1)
            - previous_dropout_risk: Previous dropout risk (0-1)
            - course_difficulty: Course difficulty (0-1)
            - study_hours: Study hours per week
            - num_failed_courses: Number of failed courses

    Returns:
        Prediction dictionary with:
            - dropout_probability: Dropout probability (0-1)
            - retention_probability: Retention probability (0-1)
            - risk_level: "low", "medium", or "high"
            - confidence: Confidence score (0-1)
            - explanation: Explanation of prediction
    """
    # Extract features with defaults
    gpa = student_data.get("gpa", 2.5)
    credits_enrolled = student_data.get("credits_enrolled", 12)
    attendance_rate = student_data.get("attendance_rate", 75.0)
    engagement_score = student_data.get("engagement_score", 0.5)
    previous_dropout_risk = student_data.get("previous_dropout_risk", 0.3)
    student_data.get("course_difficulty", 0.5)
    study_hours = student_data.get("study_hours", 10)
    num_failed_courses = student_data.get("num_failed_courses", 0)

    # Normalize values
    gpa_normalized = gpa / 4.0  # 0-1 scale
    attendance_normalized = attendance_rate / 100.0  # 0-1 scale
    min(credits_enrolled / 18.0, 1.0)  # Cap at 18 credits
    min(study_hours / 20.0, 1.0)  # Cap at 20 hours

    # Calculate risk factors (higher = more risk)
    risk_factors = {
        "gpa": (1 - gpa_normalized) * 0.30,  # Low GPA increases risk
        "attendance": (1 - attendance_normalized) * 0.25,  # Low attendance increases risk
        "engagement": (1 - engagement_score) * 0.20,  # Low engagement increases risk
        "failed_courses": min(num_failed_courses / 5.0, 1.0) * 0.15,  # Failed courses increase risk
        "previous_risk": previous_dropout_risk * 0.10,  # Previous risk is a factor
    }

    # Calculate overall risk score
    dropout_probability = sum(risk_factors.values())
    dropout_probability = min(max(dropout_probability, 0.0), 1.0)  # Clamp to 0-1

    # Determine risk level
    if dropout_probability < 0.3:
        risk_level = "low"
    elif dropout_probability < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    # Calculate retention probability
    retention_probability = 1 - dropout_probability

    # Lower confidence for rule-based (0.6 vs 0.9+ for ML)
    confidence = 0.6

    # Build explanation
    explanation = {
        "method": "rule_based",
        "message": "Using rule-based prediction (ML service unavailable)",
        "risk_factors": {
            k: round(v, 3) for k, v in risk_factors.items()
        },
        "key_indicators": [],
    }

    # Add key indicators
    if gpa < 2.0:
        explanation["key_indicators"].append("Low GPA (< 2.0)")
    if attendance_rate < 70:
        explanation["key_indicators"].append("Low attendance (< 70%)")
    if engagement_score < 0.4:
        explanation["key_indicators"].append("Low engagement (< 0.4)")
    if num_failed_courses > 2:
        explanation["key_indicators"].append(f"Multiple failed courses ({num_failed_courses})")

    logger.info(
        "Rule-based prediction generated",
        dropout_probability=dropout_probability,
        risk_level=risk_level,
        gpa=gpa,
        attendance_rate=attendance_rate,
    )

    return {
        "dropout_probability": dropout_probability,
        "retention_probability": retention_probability,
        "risk_level": risk_level,
        "confidence": confidence,
        "explanation": explanation,
    }


def rule_based_room_optimization(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based room allocation optimization.

    Falls back to this when ML service is unavailable.

    Args:
        request_data: Optimization request with:
            - sections: List of sections to allocate
            - rooms: List of available rooms

    Returns:
        Optimization result with:
            - allocation: Dictionary mapping section_id to room_id
            - metrics: Optimization metrics
            - num_sections_allocated: Number of sections allocated
            - num_violations: Number of constraint violations
    """
    sections = request_data.get("sections", [])
    rooms = request_data.get("rooms", [])

    # Simple greedy allocation: assign sections to rooms by capacity
    allocation = {}
    used_rooms = set()
    num_violations = 0

    # Sort sections by capacity requirement (descending)
    sorted_sections = sorted(
        sections,
        key=lambda s: s.get("capacity_required", 0),
        reverse=True
    )

    # Sort rooms by capacity (descending)
    sorted_rooms = sorted(
        rooms,
        key=lambda r: r.get("capacity", 0),
        reverse=True
    )

    # Greedy assignment
    for section in sorted_sections:
        section_id = section.get("id")
        required_capacity = section.get("capacity_required", 0)

        # Find first available room with sufficient capacity
        for room in sorted_rooms:
            room_id = room.get("id")
            room_capacity = room.get("capacity", 0)

            if room_id not in used_rooms and room_capacity >= required_capacity:
                allocation[section_id] = room_id
                used_rooms.add(room_id)
                break
        else:
            # No suitable room found - violation
            num_violations += 1

    num_sections_allocated = len(allocation)

    # Calculate metrics
    total_capacity_used = sum(
        next((r.get("capacity", 0) for r in rooms if r.get("id") == room_id), 0)
        for room_id in allocation.values()
    )
    total_capacity_required = sum(
        s.get("capacity_required", 0) for s in sections
    )
    utilization_rate = (
        total_capacity_required / total_capacity_used
        if total_capacity_used > 0 else 0.0
    )

    metrics = {
        "utilization_rate": utilization_rate,
        "total_capacity_used": total_capacity_used,
        "total_capacity_required": total_capacity_required,
    }

    logger.info(
        "Rule-based room optimization completed",
        num_sections_allocated=num_sections_allocated,
        num_violations=num_violations,
        utilization_rate=utilization_rate,
    )

    return {
        "allocation": allocation,
        "metrics": metrics,
        "num_sections_allocated": num_sections_allocated,
        "num_violations": num_violations,
        "explanation": {
            "method": "rule_based",
            "message": "Using greedy rule-based allocation (ML service unavailable)",
        },
    }

