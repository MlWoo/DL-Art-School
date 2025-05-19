import signal

from .logging_utils import get_root_logger


class DelayedInterrupt(object):
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals
        self.logger = get_root_logger()

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)

            def handler(s, frame):
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
                self.logger.info("Signal %s received. Delaying KeyboardInterrupt." % sig)

            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])
