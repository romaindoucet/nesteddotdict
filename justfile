# Recreate the virtual environment
env:
    rm -rf .venv uv.lock
    uv sync --all-extras

# Remove previous builds
clean:
    rm -rf dist/ *.egg-info

# Build the package
build:
    uv build

# Publish to PyPI (requires PYPI_TOKEN env variable)
publish: clean build
    uvx twine upload dist/* -u __token__ -p ${PYPI_TOKEN}
