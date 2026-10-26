# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Lightning callbacks for training TorchMD-Net potentials."""

from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

__all__ = ["EMAWeightAveraging"]


class EMAWeightAveraging(WeightAveraging):
    """Maintain an exponential moving average of model weights during training.

    The first scheduled update copies the current weights. Subsequent updates
    apply ``average = decay * average + (1 - decay) * current``. Updates follow
    optimizer steps, so gradient accumulation does not change the schedule.
    Epoch-end hooks do not trigger additional updates.

    Before the first update, validation, checkpoints, and training completion
    retain the current trained weights. Afterwards, Lightning uses the averaged
    weights for validation and inference checkpoints and restores the current
    weights to continue optimization. Checkpoints also contain the current
    weights and averaging state for resuming training. At the end of training,
    the averaged weights replace the current weights.

    This callback averages model weights independently of the loss smoothing
    controlled by ``ema_alpha_y`` and ``ema_alpha_neg_dy``.

    Parameters
    ----------
    decay : float
        Contribution of the previous average; must be strictly between 0 and 1.
    start_step : int, optional
        Zero-based optimizer-step index of the first update. For example, 1000
        skips the first 1000 optimizer steps and copies weights after step 1001.
        Default: 1000.
    update_every : int, optional
        Number of optimizer steps between updates, starting at ``start_step``.
        Must be positive. Default: 10.
    device : str or torch.device, optional
        Device holding the averaged model. Default: None (the model device).
        Use ``"cpu"`` to reduce GPU memory usage at the cost of transfers.
    use_buffers : bool, optional
        Average model buffers as well as parameters, following Lightning's
        default. If False, buffers are copied at each update. Default: True.

    Notes
    -----
    An additional model copy is kept in memory. As with Lightning's
    :class:`~lightning.pytorch.callbacks.WeightAveraging`, sharded models are
    not supported.
    """

    def __init__(
        self, decay, start_step=1000, update_every=10, device=None, use_buffers=True
    ):
        if (
            not isinstance(decay, (int, float))
            or isinstance(decay, bool)
            or not 0.0 < decay < 1.0
        ):
            raise ValueError("decay must be strictly between 0 and 1")
        if (
            not isinstance(start_step, int)
            or isinstance(start_step, bool)
            or start_step < 0
        ):
            raise ValueError("start_step must be a non-negative integer")
        if (
            not isinstance(update_every, int)
            or isinstance(update_every, bool)
            or update_every < 1
        ):
            raise ValueError("update_every must be a positive integer")
        super().__init__(
            device=device, use_buffers=use_buffers, avg_fn=get_ema_avg_fn(decay)
        )
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every

    def should_update(self, step_idx=None, epoch_idx=None):
        """Update on scheduled optimizer steps, using zero-based step indices."""
        if step_idx is None or step_idx < self.start_step:
            return False
        return (step_idx - self.start_step) % self.update_every == 0

    def _has_average(self):
        return self._average_model is not None and self._average_model.n_averaged > 0

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._has_average():
            super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._has_average():
            super().on_validation_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self._has_average():
            super().on_train_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Before EMA starts, the initial snapshot must not replace trained weights.
        # Keep n_averaged at zero so the first scheduled update still copies them.
        if self._average_model is not None and not self._has_average():
            self._average_model.module.load_state_dict(pl_module.state_dict())
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
