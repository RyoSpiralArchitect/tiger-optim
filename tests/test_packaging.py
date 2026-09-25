import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tiger_optim


def _project_metadata() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_project_version_matches_package_version():
    project = _project_metadata()["project"]
    assert project["version"] == tiger_optim.__version__


def test_project_dependencies_do_not_pin_numpy():
    project = _project_metadata()["project"]
    dependencies = [item.lower() for item in project.get("dependencies", [])]
    assert all(not item.startswith("numpy") for item in dependencies)


def test_setuptools_is_configured_for_src_layout():
    tool = _project_metadata()["tool"]["setuptools"]
    assert tool["package-dir"] == {"": "src"}


def test_optional_extras_cover_julia_and_bench_workflows():
    extras = _project_metadata()["project"]["optional-dependencies"]
    assert "julia" in extras
    assert any(dep.startswith("numpy") for dep in extras["julia"])
    assert any(dep.startswith("juliacall") for dep in extras["julia"])
    assert "bench" in extras
    assert any(dep.startswith("matplotlib") for dep in extras["bench"])
