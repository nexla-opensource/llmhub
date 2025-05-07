# Releasing to PyPI and TestPyPI

This document outlines the process for releasing the `nexla-llm-hub` package to both TestPyPI and PyPI.

## Prerequisites

1. Install the required tools:
   ```bash
   pip install build twine
   ```

2. Make sure you have accounts on:
   - [TestPyPI](https://test.pypi.org/account/register/)
   - [PyPI](https://pypi.org/account/register/)

3. Set up API tokens for both services and add them to your `~/.pypirc` file:
   ```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
   ```

## Release Process

### 1. Update Version

First, update the version in `pyproject.toml`:

```python
[project]
name = "llm_hub"
version = "x.y.z"  # Update this
```

### 2. Clean Previous Builds

Remove any old build artifacts:

```bash
rm -rf dist/ build/ *.egg-info/
```

### 3. Build the Package

```bash
python -m build
```

This will create both source distribution and wheel files in the `dist/` directory.

### 4. Upload to TestPyPI (Testing)

```bash
python -m twine upload --repository testpypi dist/*
```

### 5. Test the TestPyPI Installation

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps nexla-llm-hub
```

Verify the installation works as expected by running some basic tests:

```bash
# Test import
python -c "import llm_hub; print(llm_hub.__version__)"
```

### 6. Upload to PyPI (Production)

Once you've confirmed everything works correctly with the TestPyPI release:

```bash
python -m twine upload dist/*
```

### 7. Verify Production Installation

```bash
pip install nexla-llm-hub
```

Test the production installation to ensure it works properly.

## Release Checklist

Before releasing a new version, ensure:

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] Version is incremented according to [Semantic Versioning](https://semver.org/)
- [ ] Git tag is created for the release (e.g., `git tag v0.1.0 && git push origin v0.1.0`)

## Troubleshooting

- If you encounter errors during upload, check your `.pypirc` configuration and credentials
- For package validation errors, use `twine check dist/*` before uploading
- For installation issues, try with the `--verbose` flag: `pip install --verbose llm_hub`

## Automating Releases

Consider setting up GitHub Actions for automated releases:

1. Create a `.github/workflows/release.yml` file
2. Configure it to build and publish when a new tag is pushed
3. Store PyPI tokens as GitHub repository secrets

