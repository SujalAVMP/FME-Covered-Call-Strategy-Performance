"""Run the project's test suite in an isolated pytest environment."""

import os


def main():
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    import pytest

    raise SystemExit(pytest.main(["tests"]))


if __name__ == "__main__":
    main()
