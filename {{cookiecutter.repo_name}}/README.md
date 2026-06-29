# {{ cookiecutter.project_name }}

Generated MLOps project for {{ cookiecutter.ml_task }}.

## Stack

- API: {{ cookiecutter.api_framework }}
- Experiments: {{ cookiecutter.experiment_tracker }}
- Data versioning: {{ cookiecutter.data_versioning }}
- Orchestration: {{ cookiecutter.orchestrator }}
- Monitoring: {{ cookiecutter.monitoring_stack }}
- CI/CD: {{ cookiecutter.ci_cd }}
- Git workflow: {{ cookiecutter.git_flow }}

## Local start

```bash
pip install -e ".[dev]"
uvicorn {{ cookiecutter.package_name }}.main:app --reload
```
