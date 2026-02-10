# Features

- A list of features

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:
Successfully installed bracex-2.6 bump-my-version-1.2.6 cfgv-3.5.0 coverage-7.13.4 distlib-0.4.0 filelock-3.20.3 identify-2.6.16 nodeenv-1.10.0 pre-commit-4.5.1 promptprocessor-0.0.1 questionary-2.1.1 rich-click-1.9.7 ruff-0.15.0 virtualenv-20.36.1 wcmatch-10.1
```bash
cd promptprocessor
pip install -e .[dev]
pre-commit install
```

## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes


