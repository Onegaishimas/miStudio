miStudio Project: Naming and Coding Conventions

1. IntroductionThis document outlines the development standards, naming conventions, and best practices to be followed for all services within the miStudio project. Adhering to these conventions is crucial for maintaining code quality, consistency, and ease of collaboration as the platform grows.

2. Directory & File StructureEach microservice within the services/ directory must follow this standardized structure.services/miStudio<ServiceName>/
├── config/              # Service-specific configurations (e.g., YAML files)
├── data/                # For transient data, mounted as a volume
│   ├── input/           # Input files for processing
│   └── output/          # Generated output files
├── deployment/          # All deployment artifacts
│   ├── docker/          # Dockerfile and related files (.dockerignore)
│   └── kubernetes/      # K8s manifests (deployment.yaml, service.yaml, etc.)
├── docs/                # Service-specific documentation
├── monitoring/          # Monitoring configurations
│   ├── grafana/         # Grafana dashboard JSON models
│   └── prometheus/      # Prometheus rules (alerts.yaml, pod-monitor.yaml)
├── scripts/             # Helper shell scripts (build.sh, deploy.sh, test.sh)
├── src/                 # Main source code
│   ├── core/            # Business logic, main algorithms, and workflows
│   ├── models/          # Pydantic models for API requests/responses and data structures
│   ├── scorers/         # (For miStudioScore) Pluggable scoring modules
│   ├── utils/           # Shared utilities (logging, config loaders, etc.)
│   └── main.py          # FastAPI application entry point
└── tests/               # All tests
    ├── fixtures/        # Test data and fixtures
    ├── integration/     # End-to-end and service integration tests
    └── unit/            # Unit tests for individual modules and functions

Key Points:Service Naming: Services are named in PascalCase, prefixed with miStudio (e.g., miStudioScore).Source Code Root: All Python source code resides within the src/ directory to avoid namespace conflicts.

Separation of Concerns: The structure strictly separates code (src), deployment artifacts (deployment), configuration (config), and testing (tests).

3. Python Coding ConventionsStyle Guide: All Python code must adhere to PEP 8. Use automated formatters like black and linters like flake8 or ruff. A format.sh script should be included in each service's scripts/ directory.Naming:Modules: snake_case (e.g., activation_extractor.py).Classes: PascalCase (e.g., ScoringOrchestrator, TrainRequest).Functions & Variables: snake_case (e.g., calculate_relevance_score).Constants: UPPER_SNAKE_CASE (e.g., DEFAULT_MODEL_NAME).
   
   Type Hinting: All function signatures and variable declarations must include type hints as per PEP 484. This is critical for static analysis and clarity, especially when using FastAPI and Pydantic.
   
   Example: def load_model(model_name: str, device: str = "cpu") -> "PreTrainedModel":Docstrings: All modules, classes, and public functions must have Google-style docstrings.Example:"""Calculates the relevance score for a given feature.

Args:
    feature_data (Dict): The data object for a single feature.
    keywords (List[str]): A list of keywords to check for.

Returns:
    float: The calculated relevance score between 0 and 1.
"""
Logging: Use Python's built-in logging module. Do not use print() statements for application output. A central logging configuration should be set up in src/utils/logging_config.py and initialized in main.py.

Configuration: Avoid hardcoding values. All configuration (e.g., model names, file paths, API keys) should be managed via YAML files in the config/ directory and loaded by a utility in src/utils/.

4. API Design Conventions (FastAPI)Entry Point: The FastAPI application instance must be in src/main.py.

Request/Response Models: All API endpoints must use Pydantic models for request bodies and responses to ensure automatic validation and clear API documentation. These models should reside in src/models/api_models.py.Example:from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    features_path: str = Field(..., description="Path to the features.json file.")
    config_path: str = Field(..., description="Path to the scoring_config.json file.")
Endpoint Naming: Use plural nouns for resources and a clear verb-based action if necessary. Endpoints should be lowercase.Good: POST /score, GET /features/{feature_id}Avoid: POST /run_scoring_taskAsynchronous Operations: For long-running tasks (like training or scoring), endpoints should be asynchronous (async def). They should immediately return a job ID and a 202 Accepted status. The actual work should be handed off to a background task runner (e.g., FastAPI's BackgroundTasks or a dedicated task queue like Celery).

5. Containerization & DeploymentDockerfile: Each service must have a Dockerfile in its deployment/docker/ directory. It should be a multi-stage build to keep the final image lean.Kubernetes Manifests: All necessary Kubernetes manifests (deployment.yaml, service.yaml, configmap.yaml, etc.) must be located in deployment/kubernetes/.
   
   Resource Naming: Kubernetes resources should be named consistently using the service name as a prefix (e.g., deployment/mistudio-score, service/mistudio-score).
   
   Requirements: Python dependencies must be managed in a requirements.txt file at the root of the service directory.

6. Documentation Conventions
   
   Overall Project Docs: High-level documentation (like this document) should reside in the root /docs folder of the repository.
   
   Service-Specific Docs: Documentation pertaining to the implementation or usage of a single service should be in that service's /docs directory.
   
   File Naming: Documentation files should be named to clearly indicate their content, using hyphens to separate words.
   
   Format: <ServiceName>-<DocumentType>.md (e.g., miStudioScore-Service_Specification.md).
   
   Format: All documentation should be written in Markdown (.md).By following these conventions, we ensure that the miStudio project remains a clean, professional, and maintainable codebase that is easy for current and future developers to navigate.