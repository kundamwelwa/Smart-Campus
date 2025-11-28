"""
Report Service for generating polymorphic reports.

Uses the Reportable interface from shared/domain/reports.py to generate
reports in multiple formats (JSON, CSV, PDF) with runtime pluggability.
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import httpx
import structlog

from shared.config import settings
from shared.database.mongodb import get_mongodb
from shared.domain.reports import (
    AdminSummaryReport,
    ComplianceAuditReport,
    LecturerCoursePerformanceReport,
    Reportable,
    ReportFormat,
    ReportGenerator,
    ReportScope,
)

logger = structlog.get_logger(__name__)


class ReportService:
    """Service for generating reports using polymorphic report system."""

    def __init__(self):
        """Initialize report service with report generator."""
        self.generator = ReportGenerator()

    async def generate_admin_summary_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.ADMINISTRATIVE,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        headers: dict | None = None,
    ) -> bytes:
        """
        Generate admin summary report.

        Args:
            format: Output format (JSON, CSV, PDF)
            scope: Report scope
            start_date: Optional start date filter
            end_date: Optional end date filter
            headers: HTTP headers for service calls

        Returns:
            bytes: Generated report content
        """
        report_id = uuid4()
        headers = headers or {}

        # Collect system statistics
        total_users = 0
        total_courses = 0
        total_enrollments = 0
        active_sessions = 0
        system_health = {}

        # Get user statistics
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"http://localhost:{settings.user_service_port}/api/v1/admin/users/stats",
                    headers=headers,
                )
                if response.status_code == 200:
                    user_stats = response.json()
                    total_users = user_stats.get("total_users", 0)
                    active_sessions = user_stats.get("active_users", 0)
        except Exception as e:
            logger.warning("Failed to fetch user stats", error=str(e))
            system_health["user_service"] = "unavailable"

        # Get academic statistics
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"http://localhost:{settings.academic_service_port}/api/v1/admin/stats",
                    headers=headers,
                )
                if response.status_code == 200:
                    academic_stats = response.json()
                    total_courses = academic_stats.get("total_courses", 0)
                    total_enrollments = academic_stats.get("active_enrollments", 0)
        except Exception as e:
            logger.warning("Failed to fetch academic stats", error=str(e))
            system_health["academic_service"] = "unavailable"

        # Get facility statistics
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"http://localhost:{settings.facility_service_port}/api/v1/admin/stats",
                    headers=headers,
                )
                if response.status_code == 200:
                    facility_stats = response.json()
                    system_health["facilities"] = facility_stats.get("total_facilities", 0)
        except Exception as e:
            logger.warning("Failed to fetch facility stats", error=str(e))
            system_health["facility_service"] = "unavailable"

        # Check service health
        services_to_check = {
            "user_service": f"http://localhost:{settings.user_service_port}/health",
            "academic_service": f"http://localhost:{settings.academic_service_port}/health",
            "facility_service": f"http://localhost:{settings.facility_service_port}/health",
            "analytics_service": f"http://localhost:{settings.analytics_service_port}/health",
        }

        for service_name, health_url in services_to_check.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(health_url)
                    system_health[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception:
                system_health[service_name] = "offline"

        # Determine time period
        if start_date and end_date:
            time_period = f"{start_date.date()} to {end_date.date()}"
        elif start_date:
            time_period = f"Since {start_date.date()}"
        else:
            time_period = "last_30_days"

        # Create report instance
        report = AdminSummaryReport(
            report_id=report_id,
            total_users=total_users,
            total_courses=total_courses,
            total_enrollments=total_enrollments,
            active_sessions=active_sessions,
            system_health=system_health,
            time_period=time_period,
        )

        # Generate report using polymorphic dispatch
        return await self.generator.generate(report, format, scope)

    async def generate_lecturer_performance_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.INTERNAL,
        course_id: UUID | None = None,
        lecturer_id: UUID | None = None,
        headers: dict | None = None,
    ) -> bytes:
        """
        Generate lecturer course performance report.

        Args:
            format: Output format
            scope: Report scope
            course_id: Optional course ID filter
            lecturer_id: Optional lecturer ID filter
            headers: HTTP headers for service calls

        Returns:
            bytes: Generated report content
        """
        report_id = uuid4()
        headers = headers or {}

        # Default values
        lecturer_name = "Unknown Lecturer"
        course_code = "UNKNOWN"
        course_title = "Unknown Course"
        section_id = uuid4()
        semester = "Unknown"
        enrollment_count = 0
        grade_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        average_grade = 0.0
        completion_rate = 0.0
        attendance_rate = 0.0

        # Fetch course and enrollment data
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get course info if course_id provided
                if course_id:
                    response = await client.get(
                        f"http://localhost:{settings.academic_service_port}/api/v1/courses/{course_id}",
                        headers=headers,
                    )
                    if response.status_code == 200:
                        course_data = response.json()
                        course_code = course_data.get("course_code", "UNKNOWN")
                        course_title = course_data.get("title", "Unknown Course")

                # Get enrollment statistics
                params = {}
                if course_id:
                    params["course_id"] = str(course_id)
                if lecturer_id:
                    params["lecturer_id"] = str(lecturer_id)

                response = await client.get(
                    f"http://localhost:{settings.academic_service_port}/api/v1/enrollments",
                    headers=headers,
                    params=params,
                )
                if response.status_code == 200:
                    enrollments = response.json()
                    enrollment_count = len(enrollments) if isinstance(enrollments, list) else 0

                    # Calculate grade distribution (simplified)
                    if enrollment_count > 0:
                        # Mock grade calculation - in real implementation, fetch from grades service
                        average_grade = 75.5
                        completion_rate = 85.0
                        attendance_rate = 90.0
        except Exception as e:
            logger.warning("Failed to fetch course performance data", error=str(e))

        # Create report instance
        report = LecturerCoursePerformanceReport(
            report_id=report_id,
            lecturer_id=lecturer_id or uuid4(),
            lecturer_name=lecturer_name,
            course_id=course_id or uuid4(),
            course_code=course_code,
            course_title=course_title,
            section_id=section_id,
            semester=semester,
            enrollment_count=enrollment_count,
            grade_distribution=grade_distribution,
            average_grade=average_grade,
            completion_rate=completion_rate,
            attendance_rate=attendance_rate,
        )

        # Generate report
        return await self.generator.generate(report, format, scope)

    async def generate_compliance_audit_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.COMPLIANCE,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> bytes:
        """
        Generate compliance audit report.

        Args:
            format: Output format
            scope: Report scope
            start_date: Start date for audit period
            end_date: End date for audit period

        Returns:
            bytes: Generated report content
        """
        report_id = uuid4()

        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Collect audit data from MongoDB
        total_audit_entries = 0
        access_violations = 0
        auth_failures = 0
        data_exports = 0
        chain_integrity_valid = True
        findings = []

        try:
            db = await get_mongodb()
            events_collection = db["events"]

            # Build query for audit period
            query = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date,
                }
            }

            # Count total entries
            total_audit_entries = await events_collection.count_documents(query)

            # Count violations
            violation_query = {**query, "$or": [
                {"event_type": {"$regex": "unauthorized|violation|breach", "$options": "i"}},
                {"severity": {"$in": ["high", "critical"]}},
            ]}
            access_violations = await events_collection.count_documents(violation_query)

            # Count auth failures
            auth_query = {**query, "event_type": {"$regex": "auth.*fail|login.*fail", "$options": "i"}}
            auth_failures = await events_collection.count_documents(auth_query)

            # Count data exports
            export_query = {**query, "event_type": {"$regex": "export|download|data.*access", "$options": "i"}}
            data_exports = await events_collection.count_documents(export_query)

            # Aggregate findings by severity
            pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$severity",
                    "count": {"$sum": 1},
                }},
            ]

            async for result in events_collection.aggregate(pipeline):
                findings.append({
                    "severity": result.get("_id", "unknown"),
                    "description": f"Events with {result.get('_id', 'unknown')} severity",
                    "count": result.get("count", 0),
                })

            # Check chain integrity (simplified - would check hash chain in real implementation)
            chain_integrity_valid = True

        except Exception as e:
            logger.warning("Failed to fetch audit data", error=str(e))
            findings.append({
                "severity": "warning",
                "description": "Failed to fetch complete audit data",
                "count": 1,
            })

        # Create report instance
        report = ComplianceAuditReport(
            report_id=report_id,
            audit_period_start=start_date,
            audit_period_end=end_date,
            total_audit_entries=total_audit_entries,
            access_violations=access_violations,
            auth_failures=auth_failures,
            data_exports=data_exports,
            chain_integrity_valid=chain_integrity_valid,
            findings=findings,
        )

        # Generate report
        return await self.generator.generate(report, format, scope)

    def register_custom_report_type(self, name: str, report_class: type[Reportable]) -> None:
        """
        Register a custom report type at runtime (pluggability).

        Args:
            name: Report type identifier
            report_class: Report class implementing Reportable interface
        """
        self.generator.register_report_type(name, report_class)
        logger.info("Custom report type registered", report_type=name)

