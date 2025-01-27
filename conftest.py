import pytest

class Secret:
    """
    Taken from: https://stackoverflow.com/a/67393351
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Secret(********)"

    def __str___(self):
        return "*******"


def pytest_addoption(parser):
    parser.addoption("--token", action="store", default=None)


@pytest.fixture
def token(request):
    return Secret(request.config.option.token)
