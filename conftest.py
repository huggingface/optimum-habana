import json
import logging
import operator
import os
import sys
from pathlib import Path

import pytest


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

    def assertEqual(self, context=[], **kwargs):
        self.assertRef(operator.eq, context, **kwargs)


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
    parser.addoption(
        "--device",
        "--device-context",
        action="store",
        default=None,
        help=(
            "Used to enable device specific test configurations and baselines."
            " If unspecified, the default is to auto-detect the device."
        ),
    )


@pytest.fixture
def token(request):
    return Secret(request.config.option.token)


def pytest_configure(config):
    name = ""
    try:
        from optimum.habana.utils import get_device_name

        name = get_device_name()

        # get_device_name() returns `gaudi` for G1
        if "gaudi" == name:
            # use "gaudi1" since this is used in tests, baselines, etc.
            name = "gaudi1"
    except ValueError:
        pass  # ignore unsupported device, we'll handle it in sessionstart
    finally:
        config.stash["physical-device"] = name


def pytest_sessionstart(session):
    session.stash["baseline"] = Baseline(session)

    # User command-line option takes highest priority
    if session.config.option.device is not None:
        device = str(session.config.option.device).strip().lower()
    # Otherwise, use physical device (auto-detected)
    else:
        device = session.config.stash["physical-device"]

    if not device:
        raise RuntimeError("Expected a device context but did not detect one.")

    from tests import utils

    utils.OH_DEVICE_CONTEXT = device
    session.config.stash["device-context"] = device

    # WA: delete the imported top-level tests module so we don't overshadow
    # tests/transformers/tests module.
    # This fixes python -m pytest tests/transformers/tests/models/ -s -v
    del sys.modules["tests"]


def pytest_report_header(config):
    header = []
    if "GAUDI2_CI" in os.environ:
        del os.environ["GAUDI2_CI"]  # prevent someone from trying to use it in tests
        header.append("\n!!!!!!!!!!!!!!! NOTICE !!!!!! NOTICE !!!!!!!!!!!!!!!!!!!!!!")
        header.append("!! GAUDI2_CI environment variable has been discontinued. !!")
        header.append("!! The CI device context will be auto-detected or can be !!")
        header.append("!! overridden with '--device-context' option. See --help !!")
        header.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    header.append(f" device context: {config.stash['device-context']}")
    header.append(f"physical device: {config.stash['physical-device'] or None}")

    if config.stash["device-context"] != config.stash["physical-device"]:
        header.append("\nBEWARE: The 'device context' != 'physical-device'.")
        header.append("BEWARE: It is assumed you know what you are doing.\n")

    return header


def pytest_sessionfinish(session):
    session.stash["baseline"].finalize()


@pytest.fixture
def baseline(request):
    return BaselineRequest(request)
