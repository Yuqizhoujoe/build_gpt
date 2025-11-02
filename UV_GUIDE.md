# UV Usage Guide

`uv` is a fast Python package installer and resolver written in Rust. It's a drop-in replacement for `pip` and `venv` that's significantly faster and more reliable.

**Key features:**

- ⚡ Much faster than pip (written in Rust)
- 🔧 Works with existing Python projects
- 📦 Manages virtual environments
- ✅ Compatible with `requirements.txt` and `pyproject.toml`
- 🎯 No activation needed - use `uv run` to run commands in isolation

---

## Quick Start

### Workflow 1: Starting a New Project (from `pyproject.toml`)

**When:** Cloning a project or starting fresh with an existing `pyproject.toml`

```bash
# One command does everything:
# - Creates venv (if doesn't exist)
# - Installs all dependencies from pyproject.toml
# - Updates uv.lock for reproducible builds
uv sync

# Then run your script (no activation needed)
uv run python script.py
```

### Workflow 2: Adding Dependencies (updating `pyproject.toml`)

**When:** Adding new packages to an existing project

```bash
# Add dependencies (installs AND updates pyproject.toml)
uv add torch numpy pandas

# Add dev dependencies
uv add --dev pytest black ruff

# Run your script
uv run python script.py
```

### Workflow 3: Using `requirements.txt`

```bash
# 1. Create venv
uv venv

# 2. Install from requirements.txt
uv pip install -r requirements.txt

# 3. Run your script (no activation needed)
uv run python script.py
```

### Workflow 3: Traditional (with activation)

```bash
# 1. Create and activate venv
uv venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
# or
uv pip install -r requirements.txt

# 3. Work normally
python script.py

# 4. Deactivate when done
deactivate
```

---

## Package Management

### `uv add` (Recommended)

**Use when:** You want to track dependencies in `pyproject.toml` (recommended for projects)

```bash
# Add single package
uv add torch

# Add multiple packages
uv add numpy pandas matplotlib

# Add with version
uv add "torch>=2.0.0"

# Add dev dependencies
uv add --dev pytest black ruff

# Add from requirements.txt
uv add --requirements requirements.txt
```

**Benefits:**

- ✅ Automatically updates `pyproject.toml`
- ✅ Updates `uv.lock` for reproducible builds
- ✅ Works seamlessly with `uv sync`

### `uv pip install` (pip-style)

**Use when:** One-off installations or working with `requirements.txt` only

```bash
# Install packages
uv pip install torch numpy

# Install from requirements.txt
uv pip install -r requirements.txt

# List installed packages
uv pip list
```

### `uv sync`

**Use when:** Starting a fresh project or installing all dependencies from `pyproject.toml`

```bash
# Installs all dependencies from pyproject.toml
# - Creates venv automatically if it doesn't exist
# - Reads pyproject.toml (and uv.lock if available)
# - Installs all dependencies including dev dependencies
uv sync

# Install without dev dependencies
uv sync --no-dev

# Install only dev dependencies
uv sync --only-dev
```

**What it does:**

- ✅ Creates virtual environment if it doesn't exist
- ✅ Installs all dependencies from `pyproject.toml`
- ✅ Uses `uv.lock` for exact versions (if available)
- ✅ Updates `uv.lock` after installation
- ✅ Installs the project itself in editable mode

---

## Virtual Environment Management

### Create Virtual Environment

```bash
# Create venv in current directory (.venv)
uv venv

# With specific Python version
uv venv --python 3.11

# With custom name
uv venv myenv
```

**Note:** Virtual environments are auto-created when you run `uv sync` or `uv add` for the first time.

### Activate/Deactivate

**Traditional (manual activation):**

```bash
# Activate
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows cmd
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Deactivate
deactivate
```

**Modern (using `uv run` - no activation needed):**

```bash
# No activation needed!
uv run python script.py
uv run pytest
uv run pip install package_name
```

**Important:** `uv run` automatically uses `.venv/`, but dependencies must be installed first. It doesn't install packages automatically.

### Check Status

```bash
# Check if venv is activated
echo $VIRTUAL_ENV

# Check which Python
which python

# Check Python version
.venv/bin/python --version

# List installed packages
uv pip list
```

### Delete Virtual Environment

```bash
# Delete .venv directory
rm -rf .venv  # macOS/Linux
# rmdir /s .venv  # Windows cmd
# Remove-Item -Recurse -Force .venv  # Windows PowerShell
```

---

## Code Quality with Ruff

### What is Ruff?

**Ruff** is an extremely fast Python linter and formatter written in Rust. It's designed to be a drop-in replacement for tools like `flake8`, `black`, `isort`, `pylint`, and more—but **10-100x faster**.

**Key features:**
- 🚀 **Blazing fast** - Written in Rust, lints Python code in milliseconds
- 🔧 **All-in-one** - Combines linting and formatting in a single tool
- 📝 **Python & Jupyter** - Works with `.py` files AND `.ipynb` notebooks
- ⚙️ **Configurable** - Settings in `pyproject.toml` apply to all files
- 🎯 **Drop-in replacement** - Compatible with existing tool configurations

### Installation

```bash
# Add ruff as a dev dependency (recommended)
uv add --dev ruff

# Or install directly
uv pip install ruff
```

### Configuration

Ruff reads settings from `pyproject.toml`. Example configuration:

```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
ignore = ["E501"]  # Ignore specific rules
```

**Note:** Ruff automatically supports `.ipynb` files—no additional configuration needed!

### Basic Usage

#### Linting (Check for errors)

```bash
# Check a single file
uv run ruff check path/to/file.py

# Check all Python files in current directory
uv run ruff check .

# Check a specific notebook
uv run ruff check tokenizer/tokenizer.ipynb

# Check all notebooks
uv run ruff check **/*.ipynb

# Auto-fix issues when possible
uv run ruff check --fix .
```

#### Formatting (Fix code style)

```bash
# Format a single file
uv run ruff format path/to/file.py

# Format all Python files in current directory
uv run ruff format .

# Format a notebook
uv run ruff format tokenizer/tokenizer.ipynb

# Format all notebooks
uv run ruff format **/*.ipynb

# Check what would be formatted (dry run)
uv run ruff format --check .
```

### Common Workflows

#### Before committing code:

```bash
# Check for errors
uv run ruff check .

# Format code
uv run ruff format .

# Or combine both
uv run ruff check . && uv run ruff format .
```

#### Working with notebooks:

```bash
# Format notebook code cells
uv run ruff format tokenizer/tokenizer.ipynb

# Check notebook for linting issues
uv run ruff check tokenizer/tokenizer.ipynb

# Fix issues automatically
uv run ruff check --fix tokenizer/tokenizer.ipynb
```

### Integration with IDE

Many IDEs (VS Code, PyCharm, etc.) have Ruff extensions that automatically:
- Show linting errors as you type
- Format on save
- Apply your `pyproject.toml` settings

### Quick Reference

| Task                              | Command                              |
| --------------------------------- | ------------------------------------ |
| **Linting**                       |                                      |
| Check all files                   | `uv run ruff check .`                |
| Check specific file               | `uv run ruff check path/to/file.py`  |
| Auto-fix issues                   | `uv run ruff check --fix .`          |
| Check notebook                    | `uv run ruff check file.ipynb`       |
| **Formatting**                    |                                      |
| Format all files                  | `uv run ruff format .`               |
| Format specific file              | `uv run ruff format path/to/file.py` |
| Check formatting (dry run)        | `uv run ruff format --check .`       |
| Format notebook                   | `uv run ruff format file.ipynb`      |

---

## Quick Reference

| Action                               | Command                              |
| ------------------------------------ | ------------------------------------ |
| **Virtual Environment**              |                                      |
| Create venv                          | `uv venv`                            |
| Activate venv                        | `source .venv/bin/activate`          |
| Deactivate venv                      | `deactivate`                         |
| Delete venv                          | `rm -rf .venv`                       |
| Check if activated                   | `echo $VIRTUAL_ENV`                  |
| **Package Management**               |                                      |
| Install all from pyproject.toml      | `uv sync`                            |
| Add package (updates pyproject.toml) | `uv add package_name`                |
| Add dev dependency                   | `uv add --dev package_name`          |
| Install package (pip-style)          | `uv pip install package_name`        |
| Install from requirements.txt        | `uv pip install -r requirements.txt` |
| List packages                        | `uv pip list`                        |
| **Running Scripts**                  |                                      |
| Run without activation               | `uv run python script.py`            |
| Run with activated venv              | `python script.py`                   |
| **Code Quality (Ruff)**              |                                      |
| Check code for errors                | `uv run ruff check .`                |
| Auto-fix linting issues              | `uv run ruff check --fix .`          |
| Format code                          | `uv run ruff format .`               |
| Check formatting (dry run)           | `uv run ruff format --check .`       |
| Format/check notebook                | `uv run ruff format file.ipynb`      |

---

## Common Issues

**Problem:** `ModuleNotFoundError` when using `uv run`

**Solution:** Dependencies must be installed before running scripts:

```bash
# Check what's installed
uv pip list

# Install missing packages
uv pip install missing_package_name
# or
uv add missing_package_name

# Then run again
uv run python script.py
```
