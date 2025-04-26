import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        # Skip integration tests unless --integration is specified
        skip_integration = pytest.mark.skip(reason="Use --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration) 