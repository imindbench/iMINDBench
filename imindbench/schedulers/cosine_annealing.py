from torch.optim.lr_scheduler import CosineAnnealingLR

from .base_scheduler import BaseScheduler


class CosineAnnealing(BaseScheduler):
    """
    Cosine Annealing scheduler.

    Ported from DIVER-NeuroProbe implementation.
    """

    def __init__(self, cfg, optim):
        """
        Initialize CosineAnnealing scheduler.

        Args:
            cfg: Scheduler configuration dict with:
                - total_steps: Total number of training steps
                - eta_min_factor: Minimum LR as a factor of the initial LR
            optim: PyTorch optimizer
        """
        super().__init__()
        self.cfg = cfg
        total_steps = int(cfg.get("total_steps", 0))
        if total_steps <= 0:
            raise ValueError("cosine_annealing requires scheduler.total_steps > 0.")
        eta_min_factor = float(cfg.get("eta_min_factor", 0.0))
        if eta_min_factor < 0.0:
            raise ValueError("scheduler.eta_min_factor must be >= 0.")
        base_lr = float(optim.param_groups[0]["lr"])

        scheduler = CosineAnnealingLR(
            optim,
            T_max=total_steps,
            eta_min=base_lr * eta_min_factor,
        )
        # This zero gradient update is needed to avoid a warning message, issue #8.
        optim.zero_grad()
        optim.step()
        self.scheduler = scheduler

    def step(self, loss=None):
        """
        Step the scheduler.

        Args:
            loss: Optional loss value (not used for RampUp, but kept for compatibility)
        """
        _ = loss
        self.scheduler.step()
