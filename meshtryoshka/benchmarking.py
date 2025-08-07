import os
from collections import defaultdict
from contextlib import contextmanager
from time import time
from typing import Generator

import torch


class TimeRecorder:
    def __init__(self):
        self.times = defaultdict(list)
        self.is_benchmarking = bool(os.environ.get("BENCHMARK", False))

    @contextmanager
    def record(self, tag: str) -> Generator:
        try:
            if self.is_benchmarking:
                torch.cuda.synchronize()
                start_time = time()
            yield
        finally:
            if self.is_benchmarking:
                torch.cuda.synchronize()
                self.times[tag].append(time() - start_time)

    def state_dict(self) -> dict[str, list[float]]:
        return dict(self.times)

    def load_state_dict(self, state_dict: dict[str, list[float]]):
        self.times = defaultdict(list, state_dict)
