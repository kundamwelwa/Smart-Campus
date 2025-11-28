"""
Assignment and submission endpoints for auto-graded coursework.

Lecturer:
- Create/list assignments for their sections
- View submissions
- Approve auto-graded submissions (writes final grade)

Student:
- List assignments for enrolled sections
- Submit answers
"""

import json
import random
from datetime import date, datetime
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import (
    AnswerModel,
    AssignmentModel,
    AssignmentQuestionModel,
    CourseModel,
    EnrollmentModel,
    GradeModel,
    QuestionModel,
    SectionModel,
    SubmissionModel,
)
from shared.config import settings
from shared.database import get_db

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/assignments", tags=["assignments"])


async def get_current_user_id(authorization: str | None = Header(None)) -> UUID:
    """Decode JWT and return user ID (no role check here)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        from jose import jwt

        token = authorization.replace("Bearer ", "").strip()
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return UUID(payload.get("sub"))
    except Exception as e:
        logger.error("JWT validation failed in assignments API", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


class AssignmentCreateRequest(BaseModel):
    section_id: UUID
    title: str
    description: str | None = None
    type: str = "auto"
    due_date: datetime
    total_points: int = 100
    external_task_id: str | None = None


class AssignmentResponse(BaseModel):
    id: UUID
    course_id: UUID
    section_id: UUID
    title: str
    description: str | None
    type: str
    due_date: datetime
    total_points: int
    external_task_id: str | None
    created_by_lecturer_id: UUID
    status: str

    class Config:
        orm_mode = True


class SubmissionCreateRequest(BaseModel):
    answer: dict  # generic JSON payload describing the student's attempt


class AnswerDetailResponse(BaseModel):
    """Answer detail for submission response."""
    question_id: UUID
    question_text: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    points_earned: float
    points_possible: float
    feedback: str

    class Config:
        orm_mode = True


class SubmissionResponse(BaseModel):
    id: UUID
    assignment_id: UUID
    student_id: UUID
    submitted_at: datetime
    auto_score: float | None
    auto_feedback: str | None
    lecturer_score: float | None
    lecturer_feedback: str | None
    status: str
    answer_details: list[AnswerDetailResponse] | None = None

    class Config:
        orm_mode = True


class QuestionResponse(BaseModel):
    id: UUID
    question_type: str
    question_text: str
    question_format: str
    options: dict | None = None
    correct_answer: str | None = None
    points: int
    course_id: UUID | None = None

    class Config:
        orm_mode = True


class AnswerSubmission(BaseModel):
    question_id: UUID
    answer_text: str


class SubmissionWithAnswers(BaseModel):
    answers: list[AnswerSubmission]


class QuestionCreateRequest(BaseModel):
    course_id: UUID | None = None  # None = rule question
    question_type: str  # "rule" or "course_content"
    question_text: str
    question_format: str  # "multiple_choice", "true_false", "short_answer"
    options: dict | None = None  # For multiple choice: {"A": "option1", "B": "option2"}
    correct_answer: str
    points: int = 1


class LinkQuestionToAssignmentRequest(BaseModel):
    question_id: UUID
    is_random: bool = False  # If True, randomly select from pool
    display_order: int = 0


class ExternalTask(BaseModel):
    """External auto-grader task / template (e.g. INGInious task)."""

    id: str
    name: str
    course_code: str | None = None
    course_title: str | None = None
    description: str | None = None


async def call_external_grader(
    assignment: AssignmentModel,
    submission: SubmissionModel,
) -> tuple[float | None, str | None]:
    """
    Call configured external auto-grading service.

    This is intentionally generic: we send assignment + submission metadata and
    expect a JSON response containing at least a numeric `score` and optional `feedback`.
    """
    if not settings.external_grader_enabled or not settings.external_grader_base_url:
        logger.info("External grader disabled, skipping auto-grading")
        return None, None

    url = settings.external_grader_base_url.rstrip("/") + "/api/grade"
    payload = {
        "external_task_id": assignment.external_task_id,
        "assignment_id": str(assignment.id),
        "section_id": str(assignment.section_id),
        "course_id": str(assignment.course_id),
        "student_id": str(submission.student_id),
        "submission_id": str(submission.id),
        "answer": submission.raw_answer,
        "total_points": assignment.total_points,
    }
    headers: dict[str, str] = {}
    if settings.external_grader_api_key:
        headers["Authorization"] = f"Bearer {settings.external_grader_api_key}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        score = float(data.get("score")) if data.get("score") is not None else None
        feedback = data.get("feedback")
        return score, feedback
    except Exception as e:
        logger.error("External auto-grader call failed", error=str(e))
        return None, None


@router.post("", response_model=AssignmentResponse)
async def create_assignment(
    body: AssignmentCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Create an assignment for a section (lecturer)."""
    # Validate section and infer course
    section_stmt = select(SectionModel).where(SectionModel.id == body.section_id)
    section_result = await db.execute(section_stmt)
    section = section_result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    assignment = AssignmentModel(
        # id is generated automatically by the AssignmentModel default (uuid4)
        course_id=section.course_id,
        section_id=section.id,
        title=body.title,
        description=body.description,
        type=body.type,
        due_date=body.due_date.date(),
        total_points=body.total_points,
        external_task_id=body.external_task_id,
        created_by_lecturer_id=user_id,
        status="active",
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)
    return assignment


@router.get("", response_model=list[AssignmentResponse])
async def list_assignments_for_lecturer(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    section_id: UUID | None = Query(None),
):
    """List assignments created by the current lecturer (optionally filtered by section)."""
    stmt = select(AssignmentModel).where(AssignmentModel.created_by_lecturer_id == user_id)
    if section_id:
        stmt = stmt.where(AssignmentModel.section_id == section_id)

    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/student", response_model=list[AssignmentResponse])
async def list_assignments_for_student(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    List assignments available to the current student based on their enrollments.
    """
    # Find active enrollments
    enroll_stmt = select(EnrollmentModel).where(
        EnrollmentModel.student_id == user_id,
        EnrollmentModel.enrollment_status == "enrolled",
    )
    enroll_result = await db.execute(enroll_stmt)
    enrollments = enroll_result.scalars().all()
    section_ids = [e.section_id for e in enrollments]
    if not section_ids:
        return []

    stmt = select(AssignmentModel).where(AssignmentModel.section_id.in_(section_ids))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/questions", response_model=list[QuestionResponse])
async def list_questions(
    course_id: UUID | None = Query(None),
    question_type: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """List questions created by the current lecturer."""
    stmt = select(QuestionModel).where(QuestionModel.created_by_lecturer_id == user_id)

    if course_id:
        stmt = stmt.where(QuestionModel.course_id == course_id)

    if question_type:
        stmt = stmt.where(QuestionModel.question_type == question_type)

    stmt = stmt.where(QuestionModel.status == "active").order_by(QuestionModel.created_at.desc())

    result = await db.execute(stmt)
    question_models = result.scalars().all()

    # Explicitly construct response to ensure all fields are included
    questions = []
    for q in question_models:
        questions.append(QuestionResponse(
            id=q.id,
            question_type=q.question_type,
            question_text=q.question_text,
            question_format=q.question_format,
            options=q.options if q.options else None,
            correct_answer=q.correct_answer,
            points=q.points,
            course_id=q.course_id,
        ))

    return questions


@router.get("/questions/{question_id}", response_model=QuestionResponse)
async def get_question(
    question_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get a specific question by ID (lecturer)."""
    stmt = select(QuestionModel).where(
        QuestionModel.id == question_id,
        QuestionModel.created_by_lecturer_id == user_id,
    )
    result = await db.execute(stmt)
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Explicitly construct response to ensure all fields are included
    return QuestionResponse(
        id=question.id,
        question_type=question.question_type,
        question_text=question.question_text,
        question_format=question.question_format,
        options=question.options if question.options else None,
        correct_answer=question.correct_answer,
        points=question.points,
        course_id=question.course_id,
    )


@router.post("/questions", response_model=QuestionResponse)
async def create_question(
    body: QuestionCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Create a question (lecturer)."""
    # Validate multiple choice questions have options
    if body.question_format == "multiple_choice" and (not body.options or len(body.options) < 2):
        raise HTTPException(
            status_code=400,
            detail="Multiple choice questions must have at least 2 options"
        )

    question = QuestionModel(
        course_id=body.course_id,
        question_type=body.question_type,
        question_text=body.question_text,
        question_format=body.question_format,
        options=body.options,
        correct_answer=body.correct_answer,
        points=body.points,
        created_by_lecturer_id=user_id,
        status="active",
    )
    db.add(question)
    await db.commit()
    await db.refresh(question)

    logger.info(
        "Question created",
        question_id=str(question.id),
        question_format=question.question_format,
        has_options=question.options is not None,
        options_count=len(question.options) if question.options else 0,
        lecturer_id=str(user_id),
    )

    # Explicitly construct response to ensure all fields are included
    return QuestionResponse(
        id=question.id,
        question_type=question.question_type,
        question_text=question.question_text,
        question_format=question.question_format,
        options=question.options if question.options else None,
        correct_answer=question.correct_answer,
        points=question.points,
        course_id=question.course_id,
    )


@router.put("/questions/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: UUID,
    body: QuestionCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Update a question (lecturer - only own questions)."""
    stmt = select(QuestionModel).where(
        QuestionModel.id == question_id,
        QuestionModel.created_by_lecturer_id == user_id,
    )
    result = await db.execute(stmt)
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found or not owned by lecturer")

    # Update fields
    question.course_id = body.course_id
    question.question_type = body.question_type
    question.question_text = body.question_text
    question.question_format = body.question_format
    question.options = body.options
    question.correct_answer = body.correct_answer
    question.points = body.points

    await db.commit()
    await db.refresh(question)
    return question


@router.delete("/questions/{question_id}", response_model=dict)
async def delete_question(
    question_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete a question (lecturer - only own questions)."""
    stmt = select(QuestionModel).where(
        QuestionModel.id == question_id,
        QuestionModel.created_by_lecturer_id == user_id,
    )
    result = await db.execute(stmt)
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found or not owned by lecturer")

    # Check if question is linked to any assignments
    link_stmt = select(AssignmentQuestionModel).where(AssignmentQuestionModel.question_id == question_id)
    link_result = await db.execute(link_stmt)
    links = link_result.scalars().all()

    if links:
        # Soft delete - mark as inactive instead of hard delete
        question.status = "archived"
        await db.commit()
        return {"message": "Question archived (linked to assignments)", "question_id": str(question_id)}
    # Hard delete if not linked
    await db.delete(question)
    await db.commit()
    return {"message": "Question deleted successfully", "question_id": str(question_id)}


@router.post("/{assignment_id}/questions/link", response_model=dict)
async def link_question_to_assignment(
    assignment_id: UUID,
    body: LinkQuestionToAssignmentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Link a question to an assignment (lecturer)."""
    # Verify assignment ownership
    assign_stmt = select(AssignmentModel).where(
        AssignmentModel.id == assignment_id,
        AssignmentModel.created_by_lecturer_id == user_id,
    )
    assign_result = await db.execute(assign_stmt)
    assignment = assign_result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not owned by lecturer")

    # Verify question exists
    question_stmt = select(QuestionModel).where(QuestionModel.id == body.question_id)
    question_result = await db.execute(question_stmt)
    question = question_result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Check if already linked
    existing_stmt = select(AssignmentQuestionModel).where(
        AssignmentQuestionModel.assignment_id == assignment_id,
        AssignmentQuestionModel.question_id == body.question_id,
    )
    existing_result = await db.execute(existing_stmt)
    if existing_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Question already linked to this assignment")

    # Link question
    link = AssignmentQuestionModel(
        assignment_id=assignment_id,
        question_id=body.question_id,
        is_random=body.is_random,
        display_order=body.display_order,
    )
    db.add(link)
    await db.commit()
    return {"message": "Question linked successfully", "assignment_question_id": str(link.id)}


@router.delete("/{assignment_id}/questions/{question_id}/unlink", response_model=dict)
async def unlink_question_from_assignment(
    assignment_id: UUID,
    question_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Unlink a question from an assignment (lecturer)."""
    # Verify assignment ownership
    assign_stmt = select(AssignmentModel).where(
        AssignmentModel.id == assignment_id,
        AssignmentModel.created_by_lecturer_id == user_id,
    )
    assign_result = await db.execute(assign_stmt)
    assignment = assign_result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not owned by lecturer")

    # Find and delete the link
    link_stmt = select(AssignmentQuestionModel).where(
        AssignmentQuestionModel.assignment_id == assignment_id,
        AssignmentQuestionModel.question_id == question_id,
    )
    link_result = await db.execute(link_stmt)
    link = link_result.scalar_one_or_none()
    if not link:
        raise HTTPException(status_code=404, detail="Question not linked to this assignment")

    await db.delete(link)
    await db.commit()
    return {"message": "Question unlinked successfully"}


@router.get("/{assignment_id}/questions", response_model=list[QuestionResponse])
async def get_assignment_questions(
    assignment_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    lecturer_view: bool = Query(False, description="If true, return all questions for lecturer view"),
):
    """
    Get questions for an assignment.

    - Student view: Returns rule questions + random course content questions (only if not due)
    - Lecturer view: Returns all linked questions (set lecturer_view=true)
    """
    # Load assignment
    stmt = select(AssignmentModel).where(AssignmentModel.id == assignment_id)
    result = await db.execute(stmt)
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Lecturer view - return all questions
    if lecturer_view:
        # Verify lecturer owns the assignment
        if assignment.created_by_lecturer_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to view this assignment")

        # Get all questions linked to this assignment
        aq_stmt = (
            select(QuestionModel, AssignmentQuestionModel)
            .join(AssignmentQuestionModel, QuestionModel.id == AssignmentQuestionModel.question_id)
            .where(
                AssignmentQuestionModel.assignment_id == assignment_id,
                QuestionModel.status == "active",
            )
            .order_by(AssignmentQuestionModel.display_order)
        )
        aq_result = await db.execute(aq_stmt)
        question_rows = aq_result.all()

        # Explicitly construct response to ensure all fields are included
        questions = []
        for q, _ in question_rows:
            questions.append(QuestionResponse(
                id=q.id,
                question_type=q.question_type,
                question_text=q.question_text,
                question_format=q.question_format,
                options=q.options if q.options else None,
                correct_answer=q.correct_answer,
                points=q.points,
                course_id=q.course_id,
            ))
        return questions

    # Student view
    # Check if student is enrolled
    enroll_stmt = select(EnrollmentModel).where(
        EnrollmentModel.student_id == user_id,
        EnrollmentModel.section_id == assignment.section_id,
        EnrollmentModel.enrollment_status == "enrolled",
    )
    enroll_result = await db.execute(enroll_stmt)
    enrollment = enroll_result.scalar_one_or_none()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this section")

    # Check if assignment is due
    today = date.today()
    if assignment.due_date < today:
        raise HTTPException(status_code=400, detail="Assignment is already due")

    # Get all questions linked to this assignment
    aq_stmt = (
        select(QuestionModel, AssignmentQuestionModel)
        .join(AssignmentQuestionModel, QuestionModel.id == AssignmentQuestionModel.question_id)
        .where(
            AssignmentQuestionModel.assignment_id == assignment_id,
            QuestionModel.status == "active",
        )
        .order_by(AssignmentQuestionModel.display_order)
    )
    aq_result = await db.execute(aq_stmt)
    question_rows = aq_result.all()

    questions: list[QuestionModel] = []
    rule_questions: list[QuestionModel] = []
    course_questions: list[QuestionModel] = []

    for question, _aq in question_rows:
        if question.question_type == "rule":
            rule_questions.append(question)
        else:
            course_questions.append(question)

    # Always include all rule questions
    questions.extend(rule_questions)

    # For course questions, if random selection is enabled, pick random subset
    # Otherwise, include all
    if course_questions:
        random_questions = [q for q, aq in question_rows if q.question_type == "course_content" and aq.is_random]
        if random_questions:
            # Pick 5 random course questions (or all if less than 5)
            num_to_pick = min(5, len(course_questions))
            questions.extend(random.sample(course_questions, num_to_pick))
        else:
            questions.extend(course_questions)

    # Explicitly construct response to ensure all fields are included, especially options
    question_responses = []
    for q in questions:
        question_responses.append(QuestionResponse(
            id=q.id,
            question_type=q.question_type,
            question_text=q.question_text,
            question_format=q.question_format,
            options=q.options if q.options else None,
            correct_answer=q.correct_answer,
            points=q.points,
            course_id=q.course_id,
        ))

    logger.info(
        "Returning questions for assignment",
        assignment_id=str(assignment_id),
        question_count=len(question_responses),
        user_id=str(user_id),
        lecturer_view=lecturer_view,
    )

    return question_responses


@router.post("/{assignment_id}/submissions", response_model=SubmissionResponse)
async def submit_assignment(
    assignment_id: UUID,
    body: SubmissionWithAnswers,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Submit answers to assignment questions (student).

    Auto-grades answers and stores them.
    """
    # Load assignment
    stmt = select(AssignmentModel).where(AssignmentModel.id == assignment_id)
    result = await db.execute(stmt)
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Check enrollment
    enroll_stmt = select(EnrollmentModel).where(
        EnrollmentModel.student_id == user_id,
        EnrollmentModel.section_id == assignment.section_id,
        EnrollmentModel.enrollment_status == "enrolled",
    )
    enroll_result = await db.execute(enroll_stmt)
    enrollment = enroll_result.scalar_one_or_none()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this section")

    # Check if already submitted
    existing_stmt = select(SubmissionModel).where(
        SubmissionModel.assignment_id == assignment_id,
        SubmissionModel.student_id == user_id,
    )
    existing_result = await db.execute(existing_stmt)
    existing = existing_result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Already submitted")

    # Create submission
    submission = SubmissionModel(
        assignment_id=assignment.id,
        student_id=user_id,
        raw_answer=json.dumps([{"question_id": str(a.question_id), "answer": a.answer_text} for a in body.answers]),
        status="submitted",
    )
    db.add(submission)
    await db.flush()

    # Load questions and auto-grade answers
    total_points = 0.0
    earned_points = 0.0
    feedback_parts: list[str] = []

    question_ids = [a.question_id for a in body.answers]
    questions_stmt = select(QuestionModel).where(QuestionModel.id.in_(question_ids))
    questions_result = await db.execute(questions_stmt)
    questions_dict = {q.id: q for q in questions_result.scalars().all()}

    answers_to_save: list[AnswerModel] = []

    for answer_submission in body.answers:
        question = questions_dict.get(answer_submission.question_id)
        if not question:
            continue

        total_points += question.points
        is_correct = False
        points_earned = 0.0

        # Auto-grade based on question format
        if question.question_format == "multiple_choice":
            is_correct = answer_submission.answer_text.strip().upper() == question.correct_answer.strip().upper()
        elif question.question_format == "true_false":
            is_correct = answer_submission.answer_text.strip().lower() == question.correct_answer.strip().lower()
        else:  # short_answer - simple text match (case-insensitive)
            is_correct = answer_submission.answer_text.strip().lower() == question.correct_answer.strip().lower()

        if is_correct:
            points_earned = question.points
            earned_points += question.points

        # Create answer record
        answer = AnswerModel(
            submission_id=submission.id,
            question_id=question.id,
            student_id=user_id,
            answer_text=answer_submission.answer_text,
            is_correct=is_correct,
            points_earned=points_earned,
            points_possible=question.points,
            auto_feedback="Correct!" if is_correct else f"Incorrect. Correct answer: {question.correct_answer}",
        )
        answers_to_save.append(answer)
        feedback_parts.append(f"Q{len(answers_to_save)}: {'✓' if is_correct else '✗'}")

    # Save all answers
    for answer in answers_to_save:
        db.add(answer)

    # Update submission with auto-grading results
    submission.auto_score = earned_points
    submission.auto_feedback = f"Score: {earned_points:.1f}/{total_points:.1f}. " + " | ".join(feedback_parts)
    submission.status = "auto_graded"

    await db.commit()
    await db.refresh(submission)

    # Build answer details for response (so students can see correct answers)
    answer_details = []
    for answer in answers_to_save:
        question = questions_dict.get(answer.question_id)
        if question:
            answer_details.append(AnswerDetailResponse(
                question_id=answer.question_id,
                question_text=question.question_text,
                student_answer=answer.answer_text,
                correct_answer=question.correct_answer,
                is_correct=answer.is_correct,
                points_earned=answer.points_earned,
                points_possible=answer.points_possible,
                feedback=answer.auto_feedback,
            ))

    # Create response with answer details
    return SubmissionResponse(
        id=submission.id,
        assignment_id=submission.assignment_id,
        student_id=submission.student_id,
        submitted_at=submission.submitted_at,
        auto_score=submission.auto_score,
        auto_feedback=submission.auto_feedback,
        lecturer_score=submission.lecturer_score,
        lecturer_feedback=submission.lecturer_feedback,
        status=submission.status,
        answer_details=answer_details,
    )



@router.get("/external-tasks", response_model=list[ExternalTask])
async def list_external_tasks(
    section_id: UUID | None = Query(
        None,
        description="Optional section context to filter external tasks by course",
    ),
    db: AsyncSession = Depends(get_db),
) -> list[ExternalTask]:
    """
    List available external auto-grader tasks (e.g. INGInious tasks).

    This endpoint is a thin proxy over the configured external grader service:
    it calls `${EXTERNAL_GRADER_BASE_URL}/api/tasks` and passes optional
    course metadata derived from the section.

    If the external grader is disabled or unreachable, an empty list is returned
    so that the frontend can gracefully fall back to manual task ID entry.
    """
    if not settings.external_grader_enabled or not settings.external_grader_base_url:
        logger.info("External grader disabled, returning empty external task list")
        return []

    course_code: str | None = None
    course_title: str | None = None

    if section_id:
        # Enrich request with course context for better filtering on the adapter side
        stmt = (
            select(SectionModel, CourseModel)
            .join(CourseModel, SectionModel.course_id == CourseModel.id)
            .where(SectionModel.id == section_id)
        )
        result = await db.execute(stmt)
        row = result.first()
        if row:
            section, course = row
            course_code = course.course_code
            course_title = course.title
        else:
            logger.warning("Section not found when listing external tasks", section_id=str(section_id))

    base_url = settings.external_grader_base_url.rstrip("/")
    url = f"{base_url}/api/tasks"

    params: dict[str, str] = {}
    if course_code:
        params["course_code"] = course_code
    if course_title:
        params["course_title"] = course_title

    headers: dict[str, str] = {}
    if settings.external_grader_api_key:
        headers["Authorization"] = f"Bearer {settings.external_grader_api_key}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        tasks: list[ExternalTask] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    tasks.append(
                        ExternalTask(
                            id=str(item.get("id") or item.get("task_id")),
                            name=str(item.get("name") or item.get("title") or item.get("id")),
                            course_code=item.get("course_code"),
                            course_title=item.get("course_title"),
                            description=item.get("description"),
                        )
                    )
                except Exception as e:
                    logger.warning("Failed to parse external task entry", error=str(e), raw=item)
        else:
            logger.warning("Unexpected external tasks response format", raw_type=type(data).__name__)

        return tasks
    except Exception as e:
        logger.error("Failed to list external auto-grader tasks", error=str(e))
        # Graceful degradation: return empty list instead of failing the request
        return []

@router.get("/{assignment_id}/submissions", response_model=list[SubmissionResponse])
async def list_submissions_for_assignment(
    assignment_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """List submissions for an assignment (lecturer view)."""
    # Ensure assignment belongs to this lecturer
    assign_stmt = select(AssignmentModel).where(
        AssignmentModel.id == assignment_id,
        AssignmentModel.created_by_lecturer_id == user_id,
    )
    assign_res = await db.execute(assign_stmt)
    assignment = assign_res.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not owned by lecturer")

    stmt = select(SubmissionModel).where(SubmissionModel.assignment_id == assignment_id)
    result = await db.execute(stmt)
    return result.scalars().all()


class SubmissionApprovalRequest(BaseModel):
    lecturer_score: float | None = None
    lecturer_feedback: str | None = None


@router.post("/submissions/{submission_id}/approve", response_model=SubmissionResponse)
async def approve_submission(
    submission_id: UUID,
    body: SubmissionApprovalRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Approve an auto-graded submission (lecturer).

    This writes a final GradeModel record which is then visible to students/admin.
    """
    # Load submission + assignment
    sub_stmt = select(SubmissionModel, AssignmentModel).join(
        AssignmentModel, SubmissionModel.assignment_id == AssignmentModel.id
    ).where(SubmissionModel.id == submission_id)
    result = await db.execute(sub_stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Submission not found")

    submission, assignment = row

    # Ensure lecturer owns the assignment
    if assignment.created_by_lecturer_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to approve this submission")

    # Determine final score: lecturer override > auto_score
    final_score = body.lecturer_score if body.lecturer_score is not None else submission.auto_score
    if final_score is None:
        raise HTTPException(status_code=400, detail="No score available to approve")

    submission.lecturer_score = final_score
    if body.lecturer_feedback is not None:
        submission.lecturer_feedback = body.lecturer_feedback
    submission.approved_by_lecturer_id = user_id
    submission.approved_at = datetime.utcnow()
    submission.status = "approved"

    # Write immutable grade record
    percentage = (final_score / assignment.total_points) * 100 if assignment.total_points > 0 else 0.0
    letter = "A" if percentage >= 80 else "B" if percentage >= 70 else "C" if percentage >= 60 else "D" if percentage >= 50 else "F"

    grade = GradeModel(
        student_id=submission.student_id,
        assessment_id=assignment.id,
        section_id=assignment.section_id,
        points_earned=str(final_score),
        total_points=str(assignment.total_points),
        percentage=str(percentage),
        letter_grade=letter,
        graded_by=user_id,
        graded_at=datetime.utcnow(),
        submitted_at=submission.submitted_at,
        is_late=False,
        feedback=submission.lecturer_feedback or submission.auto_feedback or "",
        previous_grade_id=None,
        version=1,
    )
    db.add(grade)

    # Also update EnrollmentModel snapshot fields for convenience (current grade)
    enroll_stmt = select(EnrollmentModel).where(
        EnrollmentModel.student_id == submission.student_id,
        EnrollmentModel.section_id == assignment.section_id,
    )
    enroll_res = await db.execute(enroll_stmt)
    enrollment = enroll_res.scalar_one_or_none()
    if enrollment:
        enrollment.current_grade_percentage = percentage
        enrollment.current_letter_grade = letter

    await db.commit()
    await db.refresh(submission)
    return submission


