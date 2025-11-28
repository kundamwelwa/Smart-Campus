"""
Reportable Interface and Report Generation System

Implements polymorphic report generation with multiple output formats (JSON, CSV, PDF).
Supports runtime pluggability for new report types using strategy pattern.
"""

import csv
import io
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# PDF generation (optional but implemented)
PDF_AVAILABLE = False
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    PDF_AVAILABLE = True
except ImportError:
    logger.warning("ReportLab not available - PDF generation disabled")


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"


class ReportScope(str, Enum):
    """Report scope/audience."""

    PUBLIC = "public"
    INTERNAL = "internal"
    ADMINISTRATIVE = "administrative"
    COMPLIANCE = "compliance"


class ReportMetadata(BaseModel):
    """Report metadata."""

    report_id: UUID = Field(..., description="Unique report ID")
    report_type: str = Field(..., description="Report type identifier")
    title: str = Field(..., description="Report title")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: UUID | None = Field(default=None, description="User who generated report")
    scope: ReportScope = Field(default=ReportScope.INTERNAL)
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Report generation parameters"
    )


class Reportable(ABC):
    """
    Abstract interface for reportable entities and services.

    All classes that can generate reports implement this interface.
    Supports polymorphic report generation at runtime.
    """

    @abstractmethod
    async def generate_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.INTERNAL,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate report in specified format.

        Args:
            format: Output format (JSON, CSV, PDF, HTML)
            scope: Report scope/audience
            **kwargs: Format-specific options

        Returns:
            bytes: Report content as bytes

        Raises:
            ValueError: If format not supported
        """

    @abstractmethod
    def get_report_metadata(self) -> ReportMetadata:
        """
        Get report metadata.

        Returns:
            ReportMetadata: Report metadata
        """


class AdminSummaryReport(Reportable):
    """
    Administrative summary report with system-wide statistics.

    Provides high-level overview of platform usage and health.
    """

    def __init__(
        self,
        report_id: UUID,
        total_users: int,
        total_courses: int,
        total_enrollments: int,
        active_sessions: int,
        system_health: dict[str, Any],
        time_period: str = "last_30_days",
    ):
        """
        Initialize admin summary report.

        Args:
            report_id: Report ID
            total_users: Total user count
            total_courses: Total course count
            total_enrollments: Total enrollment count
            active_sessions: Active session count
            system_health: System health metrics
            time_period: Time period for statistics
        """
        self.report_id = report_id
        self.total_users = total_users
        self.total_courses = total_courses
        self.total_enrollments = total_enrollments
        self.active_sessions = active_sessions
        self.system_health = system_health
        self.time_period = time_period
        self.generated_at = datetime.utcnow()

    async def generate_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.ADMINISTRATIVE,
        **kwargs: Any,
    ) -> bytes:
        """Generate admin summary report in specified format."""
        if format == ReportFormat.JSON:
            return self._generate_json()
        if format == ReportFormat.CSV:
            return self._generate_csv()
        if format == ReportFormat.PDF:
            return self._generate_pdf()
        raise ValueError(f"Unsupported format: {format}")

    def get_report_metadata(self) -> ReportMetadata:
        """Get report metadata."""
        return ReportMetadata(
            report_id=self.report_id,
            report_type="admin_summary",
            title="Administrative Summary Report",
            generated_at=self.generated_at,
            scope=ReportScope.ADMINISTRATIVE,
            parameters={"time_period": self.time_period},
        )

    def _generate_json(self) -> bytes:
        """Generate JSON report."""
        data = {
            "metadata": self.get_report_metadata().model_dump(),
            "summary": {
                "total_users": self.total_users,
                "total_courses": self.total_courses,
                "total_enrollments": self.total_enrollments,
                "active_sessions": self.active_sessions,
                "time_period": self.time_period,
            },
            "system_health": self.system_health,
        }
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    def _generate_csv(self) -> bytes:
        """Generate CSV report."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Administrative Summary Report"])
        writer.writerow(["Generated", self.generated_at.isoformat()])
        writer.writerow(["Time Period", self.time_period])
        writer.writerow([])

        # Summary metrics
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Users", self.total_users])
        writer.writerow(["Total Courses", self.total_courses])
        writer.writerow(["Total Enrollments", self.total_enrollments])
        writer.writerow(["Active Sessions", self.active_sessions])
        writer.writerow([])

        # System health
        writer.writerow(["System Health"])
        for key, value in self.system_health.items():
            writer.writerow([key, value])

        return output.getvalue().encode("utf-8")

    def _generate_pdf(self) -> bytes:
        """Generate PDF report."""
        if not PDF_AVAILABLE:
            raise ValueError("PDF generation not available - reportlab not installed")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph("Administrative Summary Report", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        # Metadata
        metadata_text = f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}<br/>Time Period: {self.time_period}"
        elements.append(Paragraph(metadata_text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Summary metrics table
        data = [
            ["Metric", "Value"],
            ["Total Users", str(self.total_users)],
            ["Total Courses", str(self.total_courses)],
            ["Total Enrollments", str(self.total_enrollments)],
            ["Active Sessions", str(self.active_sessions)],
        ]

        table = Table(data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        # System health
        elements.append(Paragraph("System Health", styles["Heading2"]))
        health_data = [["Component", "Status"]]
        for key, value in self.system_health.items():
            health_data.append([key, str(value)])

        health_table = Table(health_data, colWidths=[3 * inch, 2 * inch])
        health_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(health_table)

        doc.build(elements)
        return buffer.getvalue()


class LecturerCoursePerformanceReport(Reportable):
    """
    Course performance report for lecturers.

    Shows detailed statistics about course enrollment, grades, and student performance.
    """

    def __init__(
        self,
        report_id: UUID,
        lecturer_id: UUID,
        lecturer_name: str,
        course_id: UUID,
        course_code: str,
        course_title: str,
        section_id: UUID,
        semester: str,
        enrollment_count: int,
        grade_distribution: dict[str, int],
        average_grade: float,
        completion_rate: float,
        attendance_rate: float,
    ):
        """Initialize lecturer course performance report."""
        self.report_id = report_id
        self.lecturer_id = lecturer_id
        self.lecturer_name = lecturer_name
        self.course_id = course_id
        self.course_code = course_code
        self.course_title = course_title
        self.section_id = section_id
        self.semester = semester
        self.enrollment_count = enrollment_count
        self.grade_distribution = grade_distribution
        self.average_grade = average_grade
        self.completion_rate = completion_rate
        self.attendance_rate = attendance_rate
        self.generated_at = datetime.utcnow()

    async def generate_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.INTERNAL,
        **kwargs: Any,
    ) -> bytes:
        """Generate course performance report."""
        if format == ReportFormat.JSON:
            return self._generate_json()
        if format == ReportFormat.CSV:
            return self._generate_csv()
        if format == ReportFormat.PDF:
            return self._generate_pdf()
        raise ValueError(f"Unsupported format: {format}")

    def get_report_metadata(self) -> ReportMetadata:
        """Get report metadata."""
        return ReportMetadata(
            report_id=self.report_id,
            report_type="lecturer_course_performance",
            title=f"Course Performance Report - {self.course_code}",
            generated_at=self.generated_at,
            generated_by=self.lecturer_id,
            scope=ReportScope.INTERNAL,
            parameters={
                "course_id": str(self.course_id),
                "section_id": str(self.section_id),
                "semester": self.semester,
            },
        )

    def _generate_json(self) -> bytes:
        """Generate JSON report."""
        data = {
            "metadata": self.get_report_metadata().model_dump(),
            "lecturer": {
                "id": str(self.lecturer_id),
                "name": self.lecturer_name,
            },
            "course": {
                "id": str(self.course_id),
                "code": self.course_code,
                "title": self.course_title,
                "section_id": str(self.section_id),
                "semester": self.semester,
            },
            "performance": {
                "enrollment_count": self.enrollment_count,
                "grade_distribution": self.grade_distribution,
                "average_grade": self.average_grade,
                "completion_rate": self.completion_rate,
                "attendance_rate": self.attendance_rate,
            },
        }
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    def _generate_csv(self) -> bytes:
        """Generate CSV report."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Course Performance Report"])
        writer.writerow(["Lecturer", self.lecturer_name])
        writer.writerow(["Course", f"{self.course_code} - {self.course_title}"])
        writer.writerow(["Semester", self.semester])
        writer.writerow(["Generated", self.generated_at.isoformat()])
        writer.writerow([])

        writer.writerow(["Performance Metrics"])
        writer.writerow(["Enrollment Count", self.enrollment_count])
        writer.writerow(["Average Grade", f"{self.average_grade:.2f}"])
        writer.writerow(["Completion Rate", f"{self.completion_rate:.1f}%"])
        writer.writerow(["Attendance Rate", f"{self.attendance_rate:.1f}%"])
        writer.writerow([])

        writer.writerow(["Grade Distribution"])
        writer.writerow(["Grade", "Count"])
        for grade, count in sorted(self.grade_distribution.items()):
            writer.writerow([grade, count])

        return output.getvalue().encode("utf-8")

    def _generate_pdf(self) -> bytes:
        """Generate PDF report."""
        if not PDF_AVAILABLE:
            raise ValueError("PDF generation not available - reportlab not installed")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph(f"Course Performance Report<br/>{self.course_code}", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        # Course info
        info_text = f"""
        <b>Lecturer:</b> {self.lecturer_name}<br/>
        <b>Course:</b> {self.course_title}<br/>
        <b>Semester:</b> {self.semester}<br/>
        <b>Generated:</b> {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(info_text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Performance metrics
        elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
        metrics_data = [
            ["Metric", "Value"],
            ["Enrollment Count", str(self.enrollment_count)],
            ["Average Grade", f"{self.average_grade:.2f}"],
            ["Completion Rate", f"{self.completion_rate:.1f}%"],
            ["Attendance Rate", f"{self.attendance_rate:.1f}%"],
        ]

        metrics_table = Table(metrics_data, colWidths=[3 * inch, 2 * inch])
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Grade distribution
        elements.append(Paragraph("Grade Distribution", styles["Heading2"]))
        grade_data = [["Grade", "Count"]]
        for grade, count in sorted(self.grade_distribution.items()):
            grade_data.append([grade, str(count)])

        grade_table = Table(grade_data, colWidths=[2 * inch, 2 * inch])
        grade_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(grade_table)

        doc.build(elements)
        return buffer.getvalue()


class ComplianceAuditReport(Reportable):
    """
    Compliance audit report for regulatory requirements.

    Summarizes audit trail, access patterns, and compliance metrics.
    """

    def __init__(
        self,
        report_id: UUID,
        audit_period_start: datetime,
        audit_period_end: datetime,
        total_audit_entries: int,
        access_violations: int,
        auth_failures: int,
        data_exports: int,
        chain_integrity_valid: bool,
        findings: list[dict[str, Any]],
    ):
        """Initialize compliance audit report."""
        self.report_id = report_id
        self.audit_period_start = audit_period_start
        self.audit_period_end = audit_period_end
        self.total_audit_entries = total_audit_entries
        self.access_violations = access_violations
        self.auth_failures = auth_failures
        self.data_exports = data_exports
        self.chain_integrity_valid = chain_integrity_valid
        self.findings = findings
        self.generated_at = datetime.utcnow()

    async def generate_report(
        self,
        format: ReportFormat,
        scope: ReportScope = ReportScope.COMPLIANCE,
        **kwargs: Any,
    ) -> bytes:
        """Generate compliance audit report."""
        if format == ReportFormat.JSON:
            return self._generate_json()
        if format == ReportFormat.CSV:
            return self._generate_csv()
        if format == ReportFormat.PDF:
            return self._generate_pdf()
        raise ValueError(f"Unsupported format: {format}")

    def get_report_metadata(self) -> ReportMetadata:
        """Get report metadata."""
        return ReportMetadata(
            report_id=self.report_id,
            report_type="compliance_audit",
            title="Compliance Audit Report",
            generated_at=self.generated_at,
            scope=ReportScope.COMPLIANCE,
            parameters={
                "period_start": self.audit_period_start.isoformat(),
                "period_end": self.audit_period_end.isoformat(),
            },
        )

    def _generate_json(self) -> bytes:
        """Generate JSON report."""
        data = {
            "metadata": self.get_report_metadata().model_dump(),
            "audit_period": {
                "start": self.audit_period_start.isoformat(),
                "end": self.audit_period_end.isoformat(),
            },
            "summary": {
                "total_audit_entries": self.total_audit_entries,
                "access_violations": self.access_violations,
                "auth_failures": self.auth_failures,
                "data_exports": self.data_exports,
                "chain_integrity_valid": self.chain_integrity_valid,
            },
            "findings": self.findings,
        }
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    def _generate_csv(self) -> bytes:
        """Generate CSV report."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Compliance Audit Report"])
        writer.writerow(
            ["Period", f"{self.audit_period_start.date()} to {self.audit_period_end.date()}"]
        )
        writer.writerow(["Generated", self.generated_at.isoformat()])
        writer.writerow([])

        writer.writerow(["Audit Summary"])
        writer.writerow(["Total Audit Entries", self.total_audit_entries])
        writer.writerow(["Access Violations", self.access_violations])
        writer.writerow(["Authentication Failures", self.auth_failures])
        writer.writerow(["Data Exports", self.data_exports])
        writer.writerow(
            ["Chain Integrity", "VALID" if self.chain_integrity_valid else "INVALID"]
        )
        writer.writerow([])

        writer.writerow(["Findings"])
        writer.writerow(["Severity", "Description", "Count"])
        for finding in self.findings:
            writer.writerow(
                [
                    finding.get("severity", ""),
                    finding.get("description", ""),
                    finding.get("count", ""),
                ]
            )

        return output.getvalue().encode("utf-8")

    def _generate_pdf(self) -> bytes:
        """Generate PDF report."""
        if not PDF_AVAILABLE:
            raise ValueError("PDF generation not available - reportlab not installed")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph("Compliance Audit Report", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        # Period info
        period_text = f"""
        <b>Audit Period:</b> {self.audit_period_start.date()} to {self.audit_period_end.date()}<br/>
        <b>Generated:</b> {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(period_text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Summary metrics
        elements.append(Paragraph("Audit Summary", styles["Heading2"]))
        summary_data = [
            ["Metric", "Value"],
            ["Total Audit Entries", str(self.total_audit_entries)],
            ["Access Violations", str(self.access_violations)],
            ["Authentication Failures", str(self.auth_failures)],
            ["Data Exports", str(self.data_exports)],
            [
                "Chain Integrity",
                "VALID" if self.chain_integrity_valid else "INVALID",
            ],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Findings
        if self.findings:
            elements.append(Paragraph("Findings", styles["Heading2"]))
            findings_data = [["Severity", "Description", "Count"]]
            for finding in self.findings:
                findings_data.append(
                    [
                        finding.get("severity", ""),
                        finding.get("description", ""),
                        str(finding.get("count", "")),
                    ]
                )

            findings_table = Table(findings_data)
            findings_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(findings_table)

        doc.build(elements)
        return buffer.getvalue()


class ReportGenerator:
    """
    Central report generation service.

    Manages report creation and format conversion.
    """

    def __init__(self):
        """Initialize report generator."""
        self.report_types: dict[str, type[Reportable]] = {
            "admin_summary": AdminSummaryReport,
            "lecturer_course_performance": LecturerCoursePerformanceReport,
            "compliance_audit": ComplianceAuditReport,
        }

    def register_report_type(self, name: str, report_class: type[Reportable]) -> None:
        """
        Register a new report type (runtime pluggability).

        Args:
            name: Report type identifier
            report_class: Report class implementing Reportable
        """
        self.report_types[name] = report_class
        logger.info("Report type registered", report_type=name)

    async def generate(
        self,
        report: Reportable,
        format: ReportFormat,
        scope: ReportScope = ReportScope.INTERNAL,
    ) -> bytes:
        """
        Generate report using polymorphic dispatch.

        Args:
            report: Report object
            format: Output format
            scope: Report scope

        Returns:
            bytes: Generated report content
        """
        logger.info(
            "Generating report",
            report_type=report.get_report_metadata().report_type,
            format=format.value,
        )

        content = await report.generate_report(format, scope)
        logger.info("Report generated", size_bytes=len(content))
        return content

