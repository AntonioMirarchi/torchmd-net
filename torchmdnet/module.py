# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
from typing import Optional, Dict, Tuple
from lightning import LightningModule
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models.utils import dtype_mapping
from torchmdnet.loss import l1_loss, loss_class_mapping, force_loss_reductions
import torch_geometric.transforms as T


class FloatCastDatasetWrapper(T.BaseTransform):
    """A transform that casts all floating point tensors to a given dtype.
    tensors to a given dtype.
    """

    def __init__(self, dtype=torch.float64):
        super(FloatCastDatasetWrapper, self).__init__()
        self._dtype = dtype

    def forward(self, data):
        for key, value in data:
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(data, key, value.to(self._dtype))
        return data


class EnergyRefRemover(T.BaseTransform):
    """A transform that removes the atom reference energy from the energy of a
    dataset.
    """

    def __init__(self, atomref):
        super(EnergyRefRemover, self).__init__()
        self._atomref = atomref

    def forward(self, data):
        self._atomref = self._atomref.to(data.z.device).type(data.y.dtype)
        if "y" in data:
            data.y.index_add_(0, data.batch, -self._atomref[data.z])
        return data


# This wrapper is here in order to permit Lightning to serialize the loss function.
class LossFunction:
    """Serializable loss with optional per-call overrides, e.g. unreduced forces."""

    def __init__(self, loss_fn, extra_args=None):
        self.loss_fn = loss_fn
        self.extra_args = extra_args
        if self.extra_args is None:
            self.extra_args = {}

    def __call__(self, x, batch, **kwargs):
        return self.loss_fn(x, batch, **{**self.extra_args, **kwargs})


class LNNP(LightningModule):
    """
    Lightning wrapper for the Neural Network Potentials in TorchMD-Net.

    ``force_loss_reduction`` selects atom-wise (default), equal-graph, or
    inverse-square-root graph weighting for forces in all stages. Energy loss
    uses its original reduction independently. See :meth:`_compute_force_loss`.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the model.
        prior_model (torchmdnet.priors.BasePrior): A prior model to use in the model.
        mean (torch.Tensor, optional): The mean of the dataset to normalize the input.
        std (torch.Tensor, optional): The standard deviation of the dataset to normalize the input.
    """

    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False
        if "train_loss" not in hparams:
            hparams["train_loss"] = "mse_loss"
        if "train_loss_arg" not in hparams:
            hparams["train_loss_arg"] = {}
        if "force_loss_reduction" not in hparams:
            hparams["force_loss_reduction"] = "atom_mean"
        if hparams["force_loss_reduction"] not in force_loss_reductions:
            raise ValueError(
                f"Unknown force_loss_reduction: {hparams['force_loss_reduction']}. "
                f"Choose from {', '.join(force_loss_reductions)}."
            )
        loss_args = hparams["train_loss_arg"]
        if loss_args is None:
            loss_args = {}
        if not isinstance(loss_args, dict):
            raise ValueError("train_loss_arg must be a dictionary")
        if loss_args.get("reduction", "mean") not in ("mean", "sum"):
            raise ValueError(
                "train_loss_arg.reduction must be 'mean' or 'sum'. "
                "Use force_loss_reduction for graph averaging; unreduced force "
                "errors are computed internally."
            )
        if hparams["force_loss_reduction"] != "atom_mean" and any(
            loss_args.get(key) is not None for key in ("size_average", "reduce")
        ):
            raise ValueError(
                "Graph force reduction does not support the deprecated "
                "size_average/reduce loss arguments; use reduction instead."
            )
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[self.hparams.precision]
        )
        if self.hparams.remove_ref_energy:
            self.data_transform = T.Compose(
                [
                    EnergyRefRemover(self.model.prior_model[-1].initial_atomref),
                    self.data_transform,
                ]
            )

        if self.hparams.train_loss not in loss_class_mapping:
            raise ValueError(
                f"Training loss {self.hparams.train_loss} not supported. Supported losses are {list(loss_class_mapping.keys())}"
            )

        self.train_loss_fn = LossFunction(
            loss_class_mapping[self.hparams.train_loss],
            self.hparams.train_loss_arg,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_metric = getattr(self.hparams, "lr_metric", "val")
        monitor = f"{lr_metric}_total_{self.hparams.train_loss}"
        lr_scheduler = {
            "scheduler": scheduler,
            "strict": True,
            "monitor": monitor,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.model(z, pos, batch=batch, box=box, q=q, s=s, extra_args=extra_args)

    def training_step(self, batch, batch_idx):
        return self.step(
            batch, [(self.hparams.train_loss, self.train_loss_fn)], "train"
        )

    def validation_step(self, batch, batch_idx, *args):
        # If args is not empty the first (and only) element is the dataloader_idx
        # We want to test every number of epochs just for reporting, but this is not supported by Lightning.
        # Instead, we trick it by providing two validation dataloaders and interpreting the second one as test.
        # The dataloader takes care of sending the two sets only when the second one is needed.
        is_val = len(args) == 0 or (len(args) > 0 and args[0] == 0)
        if is_val:
            step_type = {
                "loss_fn_list": (
                    [("l1_loss", l1_loss)]
                    if self.hparams.train_loss != "l1_loss"
                    else []
                )
                + [(self.hparams.train_loss, self.train_loss_fn)],
                "stage": "val",
            }
        else:
            step_type = {"loss_fn_list": [("l1_loss", l1_loss)], "stage": "test"}
        return self.step(batch, **step_type)

    def test_step(self, batch, batch_idx):
        return self.step(batch, [("l1_loss", l1_loss)], "test")

    def predict_step(self, batch, batch_idx):
        batch = self.data_transform(batch)

        with torch.set_grad_enabled(self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            return self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )

    def _compute_force_loss(self, prediction, batch, loss_fn):
        r"""Return the scalar force loss and its reporting count.

        ``atom_mean`` preserves the configured loss reduction and counts force
        components. For the graph modes, first average elementwise errors over
        the atoms and Cartesian components of each graph to obtain ``l_g``.
        ``graph_mean`` returns :math:`\sum_g l_g / B`, whereas
        ``graph_mean_inv_sqrt`` returns :math:`\sum_g l_g / \sqrt{N_g} / B`.
        Both count graphs for epoch reporting. The latter divides by graph count,
        not by the sum of weights, so weighting remains active for a single graph.
        ``N_g`` counts input nodes after dataset filtering (e.g. hydrogen removal).
        """
        mode = self.hparams.force_loss_reduction
        if mode == "atom_mean":
            return loss_fn(prediction, batch.neg_dy), batch.neg_dy.numel()
        if batch.batch is None:
            raise ValueError("Graph force reduction requires batch indices")

        # Override only the force reduction; retain loss parameters such as delta.
        errors = loss_fn(prediction, batch.neg_dy, reduction="none")
        per_atom = errors.mean(dim=-1)
        num_graphs = batch.num_graphs
        sizes = torch.bincount(batch.batch, minlength=num_graphs).to(errors.dtype)
        per_graph = errors.new_zeros(num_graphs).index_add_(0, batch.batch, per_atom)
        per_graph = per_graph / sizes
        if mode == "graph_mean_inv_sqrt":
            per_graph = per_graph * sizes.rsqrt()
        return per_graph.mean(), num_graphs

    def _compute_losses(self, y, neg_y, batch, loss_fn, loss_name, stage):
        # Compute the loss for the predicted value and the negative derivative (if available)
        # Args:
        #   y: predicted value
        #   neg_y: predicted negative derivative
        #   batch: batch of data
        #   loss_fn: The loss function to compute
        #   loss_name: The name of the loss function
        # Returns:
        #   loss_y: loss for the predicted value
        #   loss_neg_y: loss for the predicted negative derivative
        loss_y, loss_neg_y = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device
        )
        if self.hparams.derivative and "neg_dy" in batch:
            loss_neg_y, count = self._compute_force_loss(neg_y, batch, loss_fn)
            self._record_loss(stage, "neg_dy", loss_name, loss_fn, loss_neg_y, count)
            loss_neg_y = self._update_loss_with_ema(
                stage, "neg_dy", loss_name, loss_neg_y
            )
        if "y" in batch:
            loss_y = loss_fn(y, batch.y)
            self._record_loss(stage, "y", loss_name, loss_fn, loss_y, batch.y.numel())
            loss_y = self._update_loss_with_ema(stage, "y", loss_name, loss_y)
        return {"y": loss_y, "neg_dy": loss_neg_y}

    def _update_loss_with_ema(self, stage, type, loss_name, loss):
        # Update the loss using an exponential moving average when applicable
        # Args:
        #   stage: stage of the training (train, val, test)
        # type: type of loss (y, neg_dy)
        #   loss_name: name of the loss function
        #   loss: loss value
        alpha = getattr(self.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1 and alpha > 0:
            ema = (
                self.ema[stage][type][loss_name]
                if loss_name in self.ema[stage][type]
                else loss.detach()
            )
            loss = alpha * loss + (1 - alpha) * ema
            self.ema[stage][type][loss_name] = loss.detach()
        return loss

    def step(self, batch, loss_fn_list, stage):
        # Run a forward pass and compute the loss for each loss function
        # If the batch contains the derivative, also compute the loss for the negative derivative
        # Args:
        #   batch: batch of data
        #   loss_fn_list: list of loss functions to compute and record (the last one is used for the total loss returned by this function)
        #   stage: stage of the training (train, val, test)
        # Returns:
        #   total_loss: sum of all losses (weighted by the loss weights) for the last loss function in the provided list
        assert len(loss_fn_list) > 0
        assert self.losses is not None
        batch = self.data_transform(batch)
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            y, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )
        if self.hparams.derivative and "y" not in batch:
            # "use" both outputs of the model's forward function but discard the first
            # to only use the negative derivative and avoid 'Expected to have finished reduction
            # in the prior iteration before starting a new one.', which otherwise get's
            # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
            neg_dy = neg_dy + y.sum() * 0
        if "y" in batch and batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
        for loss_name, loss_fn in loss_fn_list:
            step_losses = self._compute_losses(
                y, neg_dy, batch, loss_fn, loss_name, stage
            )
            total_loss = (
                step_losses["y"] * self.hparams.y_weight
                + step_losses["neg_dy"] * self.hparams.neg_dy_weight
            )
        return total_loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def _record_loss(self, stage, target, loss_name, loss_fn, loss, count):
        """Accumulate detached error sums and reduction counts before loss EMA.

        Energy and force losses may have different denominators: target elements
        or graphs. Recover the sum from a mean-reduced loss; sum-reduced losses
        already contain the sum.
        Only these small statistics are retained, not predictions or batch lists.
        """
        reduction = getattr(loss_fn, "extra_args", {}).get("reduction", "mean")
        if target == "neg_dy" and self.hparams.force_loss_reduction != "atom_mean":
            reduction = "mean"
        statistics = self.losses[stage][loss_name]
        error_sum = loss.detach().to(statistics.dtype).sum()
        if reduction == "mean":
            error_sum = error_sum * count
        index = 0 if target == "y" else 1
        statistics[index, 0] += error_sum
        statistics[index, 1] += count

    def _get_epoch_loss_dict(self, stage):
        """Reduce sums/counts across ranks, then combine energy and force metrics.

        A configured sum reduction remains a sum for energy and atom-reduced forces.
        Means use target-element counts, or graph counts for graph-reduced forces,
        rather than weighting batches or ranks equally. Missing targets contribute
        zero to the total.
        """
        result = {}
        for loss_name, statistics in self.losses[stage].items():
            statistics = statistics.clone()
            if self._trainer is not None:
                statistics = self.trainer.strategy.reduce(statistics, reduce_op="sum")
            if statistics[:, 1].sum() == 0:
                continue  # This loss/stage had no labels, e.g. an unused test loader.
            reduction = "mean"
            if loss_name == self.hparams.train_loss and stage != "test":
                reduction = self.train_loss_fn.extra_args.get("reduction", "mean")
            total = statistics.new_zeros(())
            for index, target in enumerate(("y", "neg_dy")):
                value = statistics[index, 0]
                graph_force = (
                    target == "neg_dy"
                    and self.hparams.force_loss_reduction != "atom_mean"
                )
                if reduction != "sum" or graph_force:
                    value = value / statistics[index, 1].clamp_min(1)
                weight = getattr(self.hparams, f"{target}_weight")
                if weight > 0:
                    result[f"{stage}_{target}_{loss_name}"] = value
                total = total + weight * value
            result[f"{stage}_total_{loss_name}"] = total
        return result

    def on_train_epoch_start(self):
        """Start a fresh training accumulator, independent of validation runs."""
        self._reset_losses_dict("train")

    def on_validation_epoch_start(self):
        """Discard sanity-check/previous validation statistics, including test-loader data."""
        self._reset_losses_dict("val")
        self._reset_losses_dict("test")

    def on_test_epoch_start(self):
        """Start a fresh accumulator for each standalone test run."""
        self._reset_losses_dict("test")

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
            }
            if self.trainer.optimizers:
                result_dict["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            result_dict.update(self._get_epoch_loss_dict("val"))
            # Keep training monitors available to validation-time callbacks.
            result_dict.update(self._get_epoch_loss_dict("train"))
            result_dict.update(self._get_epoch_loss_dict("test"))
            # Statistics have already been synchronized across ranks.
            self.log_dict(result_dict, sync_dist=False)

    def on_test_epoch_end(self):
        self.log_dict(self._get_epoch_loss_dict("test"), sync_dist=False)

    def on_train_epoch_end(self):
        self.log_dict(self._get_epoch_loss_dict("train"), sync_dist=False)

    def _reset_losses_dict(self, stage=None):
        """Allocate error-sum/count pairs for energy and force losses on this device."""
        if stage is None:
            self.losses = {}
        # Use double precision for long epochs where supported (MPS lacks float64).
        dtype = torch.float32 if self.device.type == "mps" else torch.float64
        for name in ("train", "val", "test") if stage is None else (stage,):
            loss_names = [self.hparams.train_loss] if name == "train" else ["l1_loss"]
            if name == "val" and self.hparams.train_loss not in loss_names:
                loss_names.append(self.hparams.train_loss)
            self.losses[name] = {
                loss_name: torch.zeros(2, 2, dtype=dtype, device=self.device)
                for loss_name in loss_names
            }

    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            for loss_type in ["y", "neg_dy"]:
                self.ema[stage][loss_type] = {}
