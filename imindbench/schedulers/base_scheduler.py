"""Minimal wrapper interface shared by custom scheduler adapters."""


class BaseScheduler:
    """Base class for learning rate schedulers."""

    def step(self, *args, **kwargs):
        """Advance the wrapped scheduler by one training step."""
        raise NotImplementedError

    def load_state_dict(self, init_state):
        """Restore scheduler state from a serialized state dict."""
        self.scheduler.load_state_dict(init_state)

    def get_state_dict(self):
        """Return the wrapped scheduler state dict."""
        return self.scheduler.state_dict()

    def get_lr(self):
        """Return the current learning rate reported by the wrapped scheduler."""
        return self.scheduler._last_lr[0]
