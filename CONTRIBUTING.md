# Contributing to Edge Vision System

Thank you for contributing! To maintain code quality, system stability, and a clean history across our IoT edge and central server architecture, all contributors must follow this workflow.

---

## 1. Contribution Flow

Our development process follows a structured path to ensure stability across distributed devices:
**Issue** -> **Branch** -> **Implementation** -> **Validation** -> **Pull Request** -> **Review** -> **Merge**

### Step 1: Issue First
Before writing any code, ensure there is an open Issue describing the bug or feature. Assign it to yourself to avoid duplicated work.

### Step 2: Branch Creation
Always create a new branch from `main`. Do not push directly to `main`.
```bash
git checkout -b <type>/<brief-description>
```

---

## 2. Branch Naming Conventions

Use lowercase and hyphens. Branches must use one of the following prefixes:

- **`fix/`**: Bug fixes (e.g., `fix/rtsp-h264-decode-error`).
- **`feature/`**: New functionalities (e.g., `feature/mqtt-tls-support`).
- **`refactor/`**: Code improvements without functional changes (e.g., `refactor/ekuiper-prediction-logic`).
- **`chore/`**: Maintenance, Docker config, or dependencies (e.g., `chore/update-grafana-dashboards`).
- **`docs/`**: Documentation updates (e.g., `docs/add-deployment-guide`).

---

## 3. Pull Request Rules

1. **One PR = One Change:** Keep PRs scoped to a specific issue or component. Do not mix Edge changes with unrelated Grafana changes unless they are part of the same feature.
2. **Review Required:** No code is merged into `main` without at least one peer review.
3. **Hardware Validation:** If your changes affect the Edge Device, they **must** be validated on a physical Raspberry Pi (or equivalent testbed) before requesting a review.
4. **CI/CD & Linting:** Ensure all configurations (Docker Compose, JSON dashboards, Python scripts) pass validation.

---

## 4. Architecture Responsibilities

When contributing, keep in mind our split architecture:
- **Edge Device:** Resource-constrained. Changes to `mediamtx`, `ekuiper`, or AI models must prioritize low latency, memory efficiency, and fault tolerance against network drops.
- **Central Server:** Focus on high availability, data persistence (InfluxDB), message brokering (MQTT), and observability (Grafana).
