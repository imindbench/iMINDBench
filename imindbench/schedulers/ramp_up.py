"""Warmup + decay scheduler adapter used by PopT-style training configs."""

from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

from .base_scheduler import BaseScheduler


class RampUp(BaseScheduler):
    """
    RampUp scheduler with warmup followed by step decay.

    Adapted from PopulationTransformer's schedulers/ramp_up.py.
    Uses GradualWarmupScheduler for warmup period, then StepLR for decay.
    Warmup implementation: https://github.com/ildoonet/pytorch-gradual-warmup-lr
    """

    def __init__(self, cfg, optim):
        """
        Initialize RampUp scheduler.

        Args:
            cfg: Scheduler configuration dict with:
                - warmup_steps: Exact number of warmup updates (optional)
                - step_size_updates: Exact StepLR decay interval in updates
                  (optional)
                - num_decays: Target number of StepLR decays across the
                  post-warmup training budget (optional)
                - warmup: Fraction of total_steps for warmup (legacy fallback)
                - total_steps: Total number of training steps (required only for
                  legacy fractional behavior)
                - gamma: Decay factor for StepLR (default 0.99)
            optim: PyTorch optimizer
        """
        super().__init__()
        self.cfg = cfg
        warmup = self._resolve_warmup_steps()
        step_size = self._resolve_step_size_updates(warmup)

        gamma = 0.99
        if "gamma" in self.cfg:
            gamma = self.cfg.gamma

        self.warmup_steps = int(warmup)
        self.step_size_updates = int(step_size)
        scheduler_steplr = StepLR(optim, step_size=int(step_size), gamma=gamma)
        scheduler_warmup = GradualWarmupScheduler(
            optim, multiplier=1, total_epoch=warmup, after_scheduler=scheduler_steplr
        )

        # This zero gradient update is needed to avoid a warning message, issue #8.
        optim.zero_grad()
        optim.step()
        self.scheduler = scheduler_warmup

    def _resolve_warmup_steps(self) -> int:
        """Resolve warmup duration in optimizer-update steps."""
        if "warmup_steps" in self.cfg:
            warmup_steps = int(self.cfg.warmup_steps)
            if warmup_steps < 0:
                raise ValueError("scheduler.warmup_steps must be >= 0.")
            return warmup_steps

        if "warmup" not in self.cfg or "total_steps" not in self.cfg:
            raise ValueError(
                "RampUp requires either scheduler.warmup_steps or both "
                "scheduler.warmup and scheduler.total_steps."
            )

        warmup = int(self.cfg.warmup * self.cfg.total_steps)
        if warmup < 0:
            raise ValueError("Resolved scheduler warmup steps must be >= 0.")
        return warmup

    def _resolve_step_size_updates(self, warmup_steps: int) -> int:
        """Resolve StepLR decay interval in optimizer-update steps."""
        if "step_size_updates" in self.cfg:
            step_size_updates = int(self.cfg.step_size_updates)
            if step_size_updates <= 0:
                raise ValueError("scheduler.step_size_updates must be > 0.")
            return step_size_updates

        if "num_decays" in self.cfg:
            if "total_steps" not in self.cfg:
                raise ValueError(
                    "RampUp requires scheduler.total_steps when "
                    "scheduler.num_decays is set."
                )
            num_decays = int(self.cfg.num_decays)
            if num_decays <= 0:
                raise ValueError("scheduler.num_decays must be > 0.")
            remaining_steps = int(self.cfg.total_steps) - int(warmup_steps)
            if remaining_steps <= 0:
                raise ValueError(
                    "scheduler.total_steps must be greater than warmup steps "
                    "when scheduler.num_decays is set."
                )
            return max(1, int((remaining_steps + num_decays - 1) / num_decays))

        if "total_steps" not in self.cfg:
            raise ValueError(
                "RampUp requires scheduler.total_steps when "
                "scheduler.step_size_updates and scheduler.num_decays are not set."
            )

        step_size = int((self.cfg.total_steps - warmup_steps) / 100)
        if step_size <= 0:
            raise ValueError(
                "Resolved scheduler step size must be > 0. "
                "Set scheduler.step_size_updates explicitly or increase "
                "scheduler.total_steps."
            )
        return step_size

    def step(self, loss=None):
        """
        Step the scheduler.

        Args:
            loss: Optional loss value (not used for RampUp, but kept for compatibility)
        """
        _ = loss
        self.scheduler.step()
