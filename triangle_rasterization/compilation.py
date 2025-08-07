import logging
import sys
from contextlib import contextmanager
from functools import cache
from typing import Callable, ParamSpec, TypeVar

import torch


@contextmanager
def redirect_output(callback: Callable[[str], None]):
    """Redirect outputs for the current process. This unfortunately doesn't redirect
    outputs from child processes.
    """
    stdout = sys.stdout
    stderr = sys.stderr

    class LoggingWriter:
        def __init__(self):
            self.buffer = []

        def write(self, message: str):
            if message.endswith("\n"):
                self.buffer.append(message.removesuffix("\n"))

                # Prevent infinite recursion by not capturing output from the callback.
                sys.stdout = stdout
                sys.stderr = stderr

                callback("".join(self.buffer))

                # Turn capturing back on.
                sys.stdout = self
                sys.stderr = self

                self.buffer = []
            else:
                self.buffer.append(message)

        def flush(self):
            pass

    handler = LoggingWriter()

    # Redirect stdout and stderr to the handler.
    sys.stderr = handler
    sys.stdout = handler
    try:
        # Run the code inside the context manager.
        yield
    finally:
        # Reset stdout and stderr.
        sys.stdout = stdout
        sys.stderr = stderr


P = ParamSpec("P")
T = TypeVar("T")


def wrap_compilation(compile: Callable[P, T]) -> Callable[P, T]:
    # Cache the compilation result.
    @cache
    def wrapped_compile(*args, **kwargs):
        # Redirect stdout and stderr to logging while compiling.
        with redirect_output(logging.info):
            return compile(*args, **kwargs)

    return wrapped_compile


def is_corrupted(device: torch.device) -> bool:
    try:
        # Attempt to allocate a tensor and do math with it.
        canary = torch.zeros(tuple(), dtype=torch.float32, device=device)
        canary = 2 * canary
        return False
    except RuntimeError:
        return True
