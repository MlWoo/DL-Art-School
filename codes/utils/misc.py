import os
import signal

from .logging_utils import get_root_logger


class DelayedInterrupt(object):
    def __init__(self, signals_to_defer):
        if not isinstance(signals_to_defer, (list, tuple)):
            signals_to_defer = [signals_to_defer]
        # Store a list of signals that this instance will manage
        self.managed_signals = list(signals_to_defer)
        self.logger = get_root_logger()

        # These will be populated in __enter__
        self.signal_received_payload = {}  # Stores (actual_signal_num, frame)
        self.original_signal_handlers = {}

    def __enter__(self):
        # Clear previous state if the context manager is somehow re-entered (not typical)
        self.signal_received_payload.clear()
        self.original_signal_handlers.clear()

        # Keep track of signals for which setup was successful
        successfully_setup_signals = []

        for sig_type in self.managed_signals:
            try:
                # 1. Store the original handler (Fix: No redundant call)
                self.original_signal_handlers[sig_type] = signal.getsignal(sig_type)
                # Initialize as not received
                self.signal_received_payload[sig_type] = None

                # 2. Define the custom handler with proper closure for 'sig_type' (Fix: Closure bug)
                def create_interrupt_handler(captured_signal_type):
                    def interrupt_handler_logic(received_signal_number, frame):
                        # 'captured_signal_type' is the signal this handler was registered for.
                        # 'received_signal_number' is the actual signal that fired.
                        self.signal_received_payload[captured_signal_type] = (received_signal_number, frame)
                        self.logger.info(
                            f"Signal {received_signal_number} (handler for {captured_signal_type}) received. "
                            "Delaying processing until context exit."
                        )

                    return interrupt_handler_logic

                # 3. Set the new handler
                signal.signal(sig_type, create_interrupt_handler(sig_type))
                successfully_setup_signals.append(sig_type)
            except (ValueError, OSError, RuntimeError) as e:  # Catch common errors for signal.signal()
                self.logger.warning(
                    f"Failed to set custom handler for signal {sig_type}: {e}. "
                    f"Original handler ({self.original_signal_handlers.get(sig_type)}) remains."
                )
                # Clean up if we stored partial info for this failed signal
                if sig_type in self.original_signal_handlers:
                    del self.original_signal_handlers[sig_type]
                if sig_type in self.signal_received_payload:
                    del self.signal_received_payload[sig_type]

        # Update managed_signals to only those that were successfully set up for __exit__ processing
        self.managed_signals = successfully_setup_signals
        return self  # Return self for use with 'as' keyword if needed

    def __exit__(self, exc_type, exc_value, traceback_obj):
        for sig_type in self.managed_signals:  # Iterate only over successfully setup signals
            original_handler = self.original_signal_handlers.get(sig_type)
            received_payload = self.signal_received_payload.get(sig_type)

            # Always restore the original handler for this signal type
            if original_handler is not None:  # Should always be true if in managed_signals
                signal.signal(sig_type, original_handler)

            if received_payload:  # If this signal was received during the 'with' block
                actual_signal_num, frame = received_payload
                self.logger.info(
                    f"Context exit: Signal {actual_signal_num} (for {sig_type}) was received. "
                    f"Processing with original handler: {original_handler}"
                )

                # Fix: Robust handling of different original handler types
                if callable(original_handler):
                    # This is where KeyboardInterrupt would be re-raised for SIGINT
                    original_handler(actual_signal_num, frame)
                elif original_handler == signal.SIG_DFL:
                    self.logger.info(
                        f"Original handler for {sig_type} was SIG_DFL. "
                        f"Re-raising signal {actual_signal_num} to trigger default OS action."
                    )
                    # Ensure the handler is SIG_DFL then raise the signal to the process.
                    # This allows the OS to perform the default action (e.g., termination).
                    signal.signal(actual_signal_num, signal.SIG_DFL)  # Re-affirm DFL just in case
                    if actual_signal_num == signal.SIGINT and os.name == "posix":
                        # For SIGINT, Python's default behavior is KeyboardInterrupt.
                        # If original_handler was SIG_DFL (less common for interactive Python),
                        # raising KeyboardInterrupt might be more Pythonic.
                        # However, os.kill is more general for any SIG_DFL.
                        raise KeyboardInterrupt()
                    else:
                        # This will likely terminate the process for signals like SIGTERM, SIGHUP etc.
                        os.kill(os.getpid(), actual_signal_num)
                elif original_handler == signal.SIG_IGN:
                    self.logger.info(
                        f"Original handler for {sig_type} was SIG_IGN. "
                        f"Signal {actual_signal_num} was received but will remain ignored as per original setup."
                    )
                # If original_handler is None (e.g., set by C, not Python), there's little to do
                # from Python other than restoring it. The signal was caught and logged.

        # Return False to indicate that if an exception occurred within the 'with' block,
        # it should propagate normally. This __exit__ does not suppress other exceptions.
        return False
