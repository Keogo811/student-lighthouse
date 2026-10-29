"""Smoke test: the package imports and exposes a version."""


def test_package_imports() -> None:
    import lighthouse

    assert lighthouse.__version__ == "0.1.0"