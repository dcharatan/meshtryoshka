import logging
import signal
import sys
from typing import Callable

import torch


def register_preemption_handler(handler: Callable[[], None]) -> Callable[[], None]:
    """Register a preemption handler. This handler is called by the function returned by
    `register_preemption_handler` if preemption has been detected.
    """
    received_signal = False

    # If a preemption signal is received, record an intent to handle preemption at the
    # next opportunity.
    def signal_handler(*_) -> None:
        nonlocal received_signal
        logging.info("Received preemption signal.")
        received_signal = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # To avoid the risk of corrupting something, we only exit
    def handle_preemption_if_necessary() -> None:
        if received_signal:
            logging.info("Calling preemption handler.")

            # Just in case, wait for all CUDA kernels to finish executing.
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)

            handler()
            sys.exit(0)

    return handle_preemption_if_necessary
