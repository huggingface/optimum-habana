import json
import logging
import os
from pathlib import Path

import pytest

import tests.utils as oh_testutils


BASELINE_DIRECTORY = Path(__file__).parent.resolve() / Path("tests") / Path("baselines") / Path("fixture")


def walk_path(path: Path):
    """
    Taken from https://stackoverflow.com/a/76236680

    Path.walk() is not available until python 3.12
    """
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    files = [f for f in path.iterdir() if f.is_file()]
    yield path, subdirs, files
    for s in subdirs:
        yield from walk_path(s)


class Baseline:
    def __init__(self, session):
        self.rebase = session.config.option.rebase
        self.references = {}

        if BASELINE_DIRECTORY.exists():
            for root, dirs, files in walk_path(BASELINE_DIRECTORY):
                for name in files:
                    with (root / name).open() as f:
                        self.references.update(json.load(f))

    def get_reference(self, addr, context=[]):
        reference = self.references.setdefault(addr, {})
        for c in context:
            reference = reference.setdefault(c, {})
        return reference

    def finalize(self):
        if self.rebase:
            # aggregate refs by test file
            refsbyfile = {}
            for case, ref in self.references.items():
                key = case.split("::")[0]
                reffile = BASELINE_DIRECTORY / Path(key).with_suffix(".json")
                refsbyfile.setdefault(reffile, {})[case] = ref

            # dump aggregated refs into their own files
            for reffile, refs in refsbyfile.items():
                reffile.parent.mkdir(parents=True, exist_ok=True)
                with reffile.open("w+") as f:
                    json.dump(refs, f, indent=2, sort_keys=True)


class BaselineRequest:
    def __init__(self, request):
        self.baseline = request.session.stash["baseline"]
        self.addr = request.node.nodeid

    def assertRef(self, compare, context=[], **kwargs):
        reference = self.baseline.get_reference(self.addr, context)
        if self.baseline.rebase:
            reference.update(**kwargs)

        for key, actual in kwargs.items():
            ref = reference.get(key, None)
            logging.getLogger().info(f"{'.'.join(context + [key])}:actual = {actual}")
            logging.getLogger().info(f"{'.'.join(context + [key])}:ref    = {ref}")
            assert compare(actual, ref)


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
    parser.addoption("--rebase", action="store_true", help="rebase baseline references from current run")
    parser.addoption("--device", action="store", default=None)


@pytest.fixture
def token(request):
    return Secret(request.config.option.token)


def pytest_sessionstart(session):
    session.stash["baseline"] = Baseline(session)

    # User command-line option takes highest priority
    if session.config.option.device is not None:
        device = str(session.config.option.device).lower()
    # User GAUDI2_CI environment variable takes second priority for backwards compatibility
    elif "GAUDI2_CI" in os.environ:
        device = "gaudi2" if os.environ["GAUDI2_CI"] == "1" else "gaudi1"
    # Try to automatically detect it
    else:
        import habana_frameworks.torch.hpu as torch_hpu

        name = torch_hpu.get_device_name().strip()
        device = "unknown" if not name else name.split()[-1].lower()

    # torch_hpu.get_device_name() returns GAUDI for G1
    if "gaudi" == device:
        # use "gaudi1" since this is used in tests, baselines, etc.
        device = "gaudi1"

    oh_testutils.OH_DEVICE_CONTEXT = device


def pytest_report_header():
    return [f"device context: {oh_testutils.OH_DEVICE_CONTEXT}"]


def pytest_sessionfinish(session):
    session.stash["baseline"].finalize()


@pytest.fixture
def baseline(request):
    return BaselineRequest(request)
