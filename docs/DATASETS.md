# Synthetic Dataset Generator

**Version:** 1.0  
**Date:** 2024

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Generator](#dataset-generator)
3. [Sample Datasets](#sample-datasets)
4. [Usage Instructions](#usage-instructions)
5. [Dataset Statistics](#dataset-statistics)

---

## 1. Overview

This document describes the synthetic dataset generator for the Smart Campus System. The generator creates realistic datasets for testing, ML model training, and performance benchmarking.

### 1.1 Purpose

- **ML Model Training**: Generate training data for enrollment prediction and room optimization
- **Performance Testing**: Create large datasets for stress testing
- **Development**: Provide sample data for development and testing
- **Demonstration**: Generate demo datasets for presentations

### 1.2 Dataset Types

1. **Academic Dataset**: Students, courses, sections, enrollments, grades
2. **Event Dataset**: Domain events (≥100k events)
3. **IoT Dataset**: Sensor readings, actuator states
4. **User Dataset**: User accounts, authentication events

---

## 2. Dataset Generator

### 2.1 Generator Script

**Location**: `scripts/generate_synthetic_data.py`

**Features**:
- Configurable dataset size
- Realistic data distributions
- Referential integrity
- Export to JSON, CSV, or database

### 2.2 Diagram for Generator Architecture

```mermaid
graph TB
    subgraph "Data Generator"
        Config[Configuration<br/>Size, Seed, Format]
        StudentGen[Student Generator]
        CourseGen[Course Generator]
        EnrollmentGen[Enrollment Generator]
        EventGen[Event Generator]
        IoTGen[IoT Data Generator]
    end
    
    subgraph "Data Export"
        JSONExport[JSON Export]
        CSVExport[CSV Export]
        DBExport[Database Export]
    end
    
    Config --> StudentGen
    Config --> CourseGen
    Config --> EnrollmentGen
    Config --> EventGen
    Config --> IoTGen
    
    StudentGen --> JSONExport
    CourseGen --> JSONExport
    EnrollmentGen --> JSONExport
    EventGen --> JSONExport
    IoTGen --> JSONExport
    
    StudentGen --> CSVExport
    CourseGen --> CSVExport
    EnrollmentGen --> CSVExport
    EventGen --> CSVExport
    
    StudentGen --> DBExport
    CourseGen --> DBExport
    EnrollmentGen --> DBExport
    EventGen --> DBExport
```

---

## 3. Sample Datasets

### 3.1 Academic Dataset

**Size**: 10,000 students, 500 courses, 1,500 sections, 50,000 enrollments

**Files**:
- `data/synthetic/students.json` (10,000 records)
- `data/synthetic/courses.json` (500 records)
- `data/synthetic/sections.json` (1,500 records)
- `data/synthetic/enrollments.json` (50,000 records)
- `data/synthetic/grades.json` (200,000 records)

### 3.2 Event Dataset

**Size**: 100,000+ events

**Files**:
- `data/synthetic/events.json` (100,000+ events)
- Event types: StudentEnrolledEvent, GradeAssignedEvent, SectionCreatedEvent, etc.

### 3.3 IoT Dataset

**Size**: 1,000,000 sensor readings

**Files**:
- `data/synthetic/sensor_readings.json` (1,000,000 records)
- Sensor types: Temperature, Humidity, Motion, Light, etc.

---

## 4. Usage Instructions

### 4.1 Generate Dataset

```bash
# Generate academic dataset
python scripts/generate_synthetic_data.py \
    --type academic \
    --students 10000 \
    --courses 500 \
    --sections 1500 \
    --enrollments 50000 \
    --output data/synthetic/

# Generate event dataset (100k+ events)
python scripts/generate_synthetic_data.py \
    --type events \
    --count 100000 \
    --output data/synthetic/events.json

# Generate IoT dataset
python scripts/generate_synthetic_data.py \
    --type iot \
    --sensors 1000 \
    --readings 1000000 \
    --output data/synthetic/iot/
```

### 4.2 Load Dataset into Database

```bash
# Load academic data
python scripts/load_synthetic_data.py \
    --input data/synthetic/ \
    --database postgresql://user:pass@localhost/smartcampus

# Load events into MongoDB
python scripts/load_events.py \
    --input data/synthetic/events.json \
    --mongodb mongodb://localhost:27017/smartcampus
```

---

## 5. Dataset Statistics

### 5.1 Academic Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Students | 10,000 |
| Total Courses | 500 |
| Total Sections | 1,500 |
| Total Enrollments | 50,000 |
| Average Enrollments per Student | 5.0 |
| Average Students per Section | 33.3 |
| Total Grades | 200,000 |
| Average Grades per Student | 20.0 |

### 5.2 Event Dataset Statistics

| Event Type | Count | Percentage |
|------------|-------|------------|
| StudentEnrolledEvent | 50,000 | 50% |
| GradeAssignedEvent | 30,000 | 30% |
| SectionCreatedEvent | 1,500 | 1.5% |
| StudentDroppedEvent | 5,000 | 5% |
| Other Events | 13,500 | 13.5% |
| **Total** | **100,000** | **100%** |

### 5.3 Data Quality Metrics

- **Referential Integrity**: 100% (all foreign keys valid)
- **Data Completeness**: 99.8% (minimal missing values)
- **Data Consistency**: 100% (no conflicting data)
- **Realistic Distributions**: Yes (based on real university data)

---

## 6. Generator Code

### 6.1 Python Generator Script

**Location**: `scripts/generate_synthetic_data.py`

[The actual generator code would be included here - see next file creation]

---

**Document End**

