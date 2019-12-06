import logging
import os
import unittest
from distutils.util import strtobool

from transformers.file_utils import _tf_available, _torch_available

logger = logging.getLogger(__name__)


def get_bool_from_env_var(name, default):
    try:
        value = os.environ[name]
    except KeyError:
        return default

    try:
        return strtobool(value)
    except ValueError:
        # More values are supported, but let's keep the message simple.
        raise ValueError("If set, %s must be yes or no." % name)


_run_slow_tests = get_bool_from_env_var("RUN_SLOW", default=False)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to yes to run them.

    """
    if not _run_slow_tests:
        test_case = unittest.skip("test is slow")(test_case)
    return test_case


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not _torch_available:
        test_case = unittest.skip("test requires PyTorch")(test_case)
    return test_case


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not _tf_available:
        test_case = unittest.skip("test requires TensorFlow")(test_case)
    return test_case


_run_tests_on_gpu = get_bool_from_env_var("USE_GPU", default=None)


if _torch_available:
    import torch

    # Set the CUDA_VISIBLE_DEVICES environment variable to select a specific
    # GPU if the machine has more than one.

    # Set the USE_GPU environment variable to "no" to prevent tests from
    # running on the GPU when it's available.

    if _run_tests_on_gpu is None:
        # USE_GPU isn't set -> detect GPU support and use it automatically.
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif _run_tests_on_gpu:
        # USE_GPU ="yes" -> detect GPU support and warn if it's unavailable.
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_device == "cpu":
            logger.warning(
                "USE_GPU is enabled but no GPU is available, "
                "tests will run on CPU anyway."
            )
    else:
        # USE_GPU="no" -> run tests on the CPU.
        torch_device = "cpu"
else:
    torch_device = None
