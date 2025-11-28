# 🎓 Argos Smart Campus – Start Here ✅

This is the **main entrypoint** for running and understanding your Argos Smart Campus platform for the assignment **"Argos: A Federated, Adaptive Smart Campus Orchestration Platform"**.

**Quick Start:** Run Docker → Install Python deps → Run `START_ALL.bat` → Open `http://localhost:5173`

---

## 📋 Table of Contents

1. [What's in this project](#1-whats-in-this-project)
2. [System Requirements](#2-system-requirements)
3. [Setup Instructions](#3-setup-instructions)
4. [Running the System](#4-running-the-system)
5. [Accessing the Application](#5-accessing-the-application)
6. [Testing](#6-testing)
7. [Documentation Links](#7-documentation-links)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What's in this project

- **Backend**: Python 3.11+, FastAPI microservices (`services/…`), shared domain models (`shared/domain`)
- **Frontend**: React + TypeScript + Vite (`frontend/`), role‑based dashboard for Student/Lecturer/Staff/Admin
- **Databases & infra**: PostgreSQL, MongoDB, Redis, Kafka – all via Docker
- **ML/Analytics**: Enrollment dropout predictor + room optimizer (with rule‑based fallback if ML deps not installed)
- **Assignment extras**: Event sourcing, Raft consensus, plugin system, audit chain, policy engine

---

## 2. System Requirements

### Prerequisites

- **Python 3.11+** (recommended; ML stack is tested there)
- **Docker Desktop** (for PostgreSQL, MongoDB, Redis, Kafka)
- **Node.js 18+** (for frontend)
- **Git** (for cloning the repository)

### Optional (for full ML features)

- **CUDA-capable GPU** (optional, for faster ML training)
- **8GB+ RAM** (recommended for ML workloads)

---

## 3. Setup Instructions

### 3.1 Clone the Repository

```bash
git clone <repository-url>
cd "Smart Campus"
```

### 3.2 Set Up Python Environment

From the project root:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**Important notes:**

- `requirements.txt` contains **all Python libraries** used in this project (core services, ML, dev tools)
- ML packages (`torch`, `ray[rllib]`) are **large** (~2–3 GB) and may take 10–30 minutes to install
- If ML packages fail to install (e.g., on Python 3.14 or older systems), **the system still works**:
  - The Analytics service will use rule-based fallbacks instead of ML models
  - All other services (API Gateway, User, Academic, etc.) work normally
- If you want to skip ML packages entirely, you can install core dependencies manually

**Dependency files:**

- `requirements.txt` – **canonical full list** (use this for complete installation)
- `requirements-quick.txt` – optional fast install (unpinned, latest versions)
- `requirements-minimal.txt` – minimal set without ML (for testing without heavy packages)
- `pyproject.toml` – authoritative list (also defines extras like `[dev]`, but `requirements.txt` mirrors it)

### 3.3 Set Up Environment Variables

Copy the example environment file and configure it:

```bash
# Windows
copy config.env.example .env

# Linux/macOS
cp config.env.example .env
```

Edit `.env` with your configuration (database passwords, API keys, etc.). See `config.env.example` for all available options.

### 3.4 Set Up Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3.5 Start Docker Infrastructure

All infra services run via Docker Compose.

From the project root:

```bash
docker compose up -d
```

This brings up:

- **PostgreSQL** (port 5432) – Main relational database
- **MongoDB** (port 27017) – Event store
- **Redis** (port 6379) – Cache & session store
- **Kafka** (port 9092) + **Zookeeper** – Event streaming

**Verify Docker services are running:**

```bash
docker compose ps
```

All services should show "Up" status.

**For more detail** (ports, troubleshooting, manual setup), see:
- `docker-compose.yml` – Docker configuration
- `RESTART_API_GATEWAY.md` – Troubleshooting guide

### 3.6 Initialize Database (First Time Only)

```bash
# With virtualenv activated
python scripts/init-db.py
```

This creates the database schema and initial data.

---

## 4. Running the System

### 4.1 Quick Start (Recommended)

After Docker is up and Python deps are installed:

**Windows:**
```powershell
.\START_ALL.bat
```

**Linux/macOS:**
```bash
# Start all services manually (see section 4.2)
```

This will:
- Start all FastAPI services (API Gateway, User, Academic, Analytics, Facility, Scheduler, Security, Consensus)
- Start the React frontend dev server

### 4.2 Manual Start (Alternative)

If you prefer manual control or need to debug:

**Terminal 1 - API Gateway:**
```bash
cd services/api_gateway
uvicorn main:app --reload --port 8000
```

**Terminal 2 - User Service:**
```bash
cd services/user_service
uvicorn main:app --reload --port 8001
```

**Terminal 3 - Academic Service:**
```bash
cd services/academic_service
uvicorn main:app --reload --port 8002
```

**Terminal 4 - Analytics Service:**
```bash
cd services/analytics_service
uvicorn main:app --reload --port 8004
```

**Terminal 5 - Frontend:**
```bash
cd frontend
npm run dev
```

**Note:** You may need to start additional services (Facility, Scheduler, Security, Consensus) depending on your use case.

### 4.3 Running ML / Analytics

The system runs **without** heavy ML dependencies by using rule‑based fallbacks.

**To enable full ML** (LSTM + PPO):

**Windows:**
```powershell
.\INSTALL_PACKAGES.bat
```

**Linux/macOS:**
```bash
pip install torch pytorch-lightning ray[rllib] scikit-learn pandas numpy shap lime
```

**Analytics service endpoints:**
- `POST /api/v1/analytics/predict/enrollment` – dropout risk prediction (with explainability)
- `POST /api/v1/analytics/optimize/rooms` – room allocation optimizer

Admin UI has an **Analytics** page that calls these via:
- `GET /api/v1/admin/ml/models`
- `POST /api/v1/analytics/predict/enrollment`

---

## 5. Accessing the Application

Once all services are running:

### Frontend Application
- **URL**: `http://localhost:5173`
- **Default Users** (if initialized):
  - Admin: `admin@argos.edu` / `admin123`
  - Lecturer: `lecturer@argos.edu` / `lecturer123`
  - Student: `student@argos.edu` / `student123`

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Service Health Checks
- **API Gateway**: `http://localhost:8000/health`
- **User Service**: `http://localhost:8001/health`
- **Academic Service**: `http://localhost:8002/health`

---

## 6. Testing

### 6.1 Run Unit Tests

With the virtualenv active:

```bash
pytest
```

**Test coverage includes:**
- Policy unit tests (`tests/test_policies.py`) – checks enrollment policies & invariants
- Analytics fallback test (`tests/test_analytics_fallback.py`) – ensures ML endpoint works even without deep ML
- API versioning tests (`tests/test_api_versioning.py`)
- Concurrency stress tests (`tests/test_concurrency_stress.py`)
- Formal verification tests (`tests/test_formal_verification.py`)
- Security penetration tests (`tests/test_penetration.py`)

### 6.2 Run Linting & Type Checking

```bash
# Linting (ruff)
ruff check shared services ml

# Type checking (mypy)
mypy shared services
```

### 6.3 CI/CD

GitHub Actions CI is configured in:
- `.github/workflows/ci.yml` – runs lint (`ruff`), type check (`mypy`), and tests on push/PR

### 6.4 Optional: Stress & Security Tests

From project root (with API Gateway + services running):

**Enrollment stress test:**
```bash
# Windows PowerShell
$env:STRESS_API_BASE="http://localhost:8000"
$env:STRESS_ACCESS_TOKEN="your-token"
$env:STRESS_STUDENT_ID="student-id"
$env:STRESS_SECTION_ID="section-id"
python -m scripts.enrollment_stress_test

# Linux/macOS
STRESS_API_BASE=http://localhost:8000 \
STRESS_ACCESS_TOKEN=... \
STRESS_STUDENT_ID=... \
STRESS_SECTION_ID=... \
python -m scripts.enrollment_stress_test
```

**Security pen‑test harness:**
```bash
# Windows PowerShell
$env:SEC_API_BASE="http://localhost:8000"
$env:SEC_ADMIN_ACCESS_TOKEN="admin-token"
$env:SEC_STUDENT_ACCESS_TOKEN="student-token"
python -m scripts.security_pentest

# Linux/macOS
SEC_API_BASE=http://localhost:8000 \
SEC_ADMIN_ACCESS_TOKEN=... \
SEC_STUDENT_ACCESS_TOKEN=... \
python -m scripts.security_pentest
```

These are **not mandatory for running** the system, but useful for producing the test/security reports required by the assignment.

---

## 7. Documentation Links

### 📚 Core Documentation

- **[README.md](./README.md)** – Project overview and quick reference
- **[RESTART_API_GATEWAY.md](./RESTART_API_GATEWAY.md)** – Troubleshooting guide for API Gateway

### 📖 Technical Documentation (`docs/` folder)

- **[docs/README.md](./docs/README.md)** – Technical documentation index
- **[docs/DESIGN_DOCUMENT.md](./docs/DESIGN_DOCUMENT.md)** – System architecture and design decisions
- **[docs/UML_DIAGRAMS.md](./docs/UML_DIAGRAMS.md)** – UML class diagrams, sequence diagrams, and system models
- **[docs/OBJECT_MODEL_SUMMARY.md](./docs/OBJECT_MODEL_SUMMARY.md)** – Domain model overview
- **[docs/CONCURRENCY_AND_API_SUMMARY.md](./docs/CONCURRENCY_AND_API_SUMMARY.md)** – Concurrency patterns and API design
- **[docs/FORMAL_VERIFICATION.md](./docs/FORMAL_VERIFICATION.md)** – Formal verification of critical invariants
- **[docs/SECURITY_AND_COMPLIANCE_SUMMARY.md](./docs/SECURITY_AND_COMPLIANCE_SUMMARY.md)** – Security measures and compliance
- **[docs/TEST_REPORT.md](./docs/TEST_REPORT.md)** – Test coverage and results
- **[docs/DATASETS.md](./docs/DATASETS.md)** – ML datasets and data generation

### 🏗️ Infrastructure Documentation

- **[infrastructure/k8s/README.md](./infrastructure/k8s/README.md)** – Kubernetes deployment guide
- **[infrastructure/terraform/README.md](./infrastructure/terraform/README.md)** – Infrastructure as Code (Terraform)
- **[infrastructure/monitoring/prometheus.yml](./infrastructure/monitoring/prometheus.yml)** – Monitoring configuration

### 💻 Frontend Documentation

- **[frontend/README.md](./frontend/README.md)** – Frontend setup and development guide

### 📝 Code Documentation

- **API Endpoints**: See Swagger UI at `http://localhost:8000/docs` when running
- **gRPC Services**: See `shared/grpc/protos/` for protocol buffer definitions
- **Event Schema**: See `shared/events/` for event definitions

---

## 8. Troubleshooting

### Common Issues

**Docker services won't start:**
```bash
# Check Docker is running
docker ps

# Restart Docker services
docker compose down
docker compose up -d

# Check logs
docker compose logs
```

**Python import errors:**
```bash
# Ensure virtualenv is activated
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Port already in use:**
- Check if services are already running: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux/macOS)
- Stop conflicting services or change ports in `config.env.example`

**Frontend won't connect to backend:**
- Verify API Gateway is running on port 8000
- Check `frontend/.env` has `VITE_API_URL=http://localhost:8000`
- Check browser console for CORS errors

**Database connection errors:**
- Verify Docker containers are running: `docker compose ps`
- Check database credentials in `.env` match `docker-compose.yml`
- Run database initialization: `python scripts/init-db.py`

**ML features not working:**
- System works without ML packages (uses fallbacks)
- To enable ML: Install packages with `INSTALL_PACKAGES.bat` or `pip install torch pytorch-lightning ray[rllib]`
- Check Analytics service logs for ML-related errors

### Getting Help

1. Check the relevant documentation file (see [Documentation Links](#7-documentation-links))
2. Review service logs in terminal output
3. Check Docker logs: `docker compose logs <service-name>`
4. Verify all prerequisites are installed correctly

---

## 🚀 Quick Reference

**If you only remember one thing:** 

1. **Start Docker**: `docker compose up -d`
2. **Activate venv**: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS)
3. **Install deps**: `pip install -r requirements.txt`
4. **Run system**: `.\START_ALL.bat` (Windows) or start services manually
5. **Open browser**: `http://localhost:5173`

Everything else is documented in the links above! 📚

---

## 📞 Next Steps

- Explore the [API documentation](http://localhost:8000/docs) when running
- Review [Design Document](./docs/DESIGN_DOCUMENT.md) for architecture details
- Check [UML Diagrams](./docs/UML_DIAGRAMS.md) for system models
- Read [Test Report](./docs/TEST_REPORT.md) for test coverage

---

## 👤 Author

**Kellycode - 2464**  
📧 [kundamwelwa7@gmail.com](mailto:kundamwelwa7@gmail.com)

---

**Happy coding! 🎉**
