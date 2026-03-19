# demo-hyformer

[![Release](https://img.shields.io/github/v/release/your-github-handle/demo-hyformer)](https://img.shields.io/github/v/release/your-github-handle/demo-hyformer)
[![Build status](https://img.shields.io/github/actions/workflow/status/your-github-handle/demo-hyformer/main.yml?branch=main)](https://github.com/your-github-handle/demo-hyformer/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/your-github-handle/demo-hyformer/branch/main/graph/badge.svg)](https://codecov.io/gh/your-github-handle/demo-hyformer)
[![Commit activity](https://img.shields.io/github/commit-activity/m/your-github-handle/demo-hyformer)](https://img.shields.io/github/commit-activity/m/your-github-handle/demo-hyformer)
[![License](https://img.shields.io/github/license/your-github-handle/demo-hyformer)](https://img.shields.io/github/license/your-github-handle/demo-hyformer)

This is a template repository for Pytorch Lightning projects that use uv for their dependency management.

- **Github repository**: <https://github.com/your-github-handle/demo-hyformer/>
- **Documentation** <https://your-github-handle.github.io/demo-hyformer/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:your-github-handle/demo-hyformer.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Verification

To verify your installation is working correctly:

```bash
# Run code quality checks
make check

# Run tests
make test

# Check if PyTorch can detect your GPU (if available)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Try your first train with
uv run python -m demo_hyformer.scripts.train
```

Since it create the virtual environment using uv, please use uv run for all your python script like `uv run python` or just `source ./.venv/bin/activate` to enter the environment first and then run python command.

### 4. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 5. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://foreveryounggithub.github.io/cookiecutter-pytorch-lightning/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://foreveryounggithub.github.io/cookiecutter-pytorch-lightning/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://foreveryounggithub.github.io/cookiecutter-pytorch-lightning/features/codecov/).

## Releasing a new version



---

Repository initiated with [foreverYoungGitHub/cookiecutter-pytorch-lightning](https://github.com/foreverYoungGitHub/cookiecutter-pytorch-lightning).
