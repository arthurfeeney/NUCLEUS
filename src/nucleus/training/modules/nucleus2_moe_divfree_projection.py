from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_augmentation import (
    FlipAugmentation,
    ForecastAugmentation,
    GaussianNoiseAugmentation,
)
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.models.nucleus2_moe_divfree_projection import Nucleus2MoEDivFreeProjection, Nucleus2MoEDivFreeProjectionOutput
from nucleus.physics.coordinates import CellCenteredGrid
from nucleus.physics.div_projection import div_projection
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.training.loop import Metrics, StepOutput
from nucleus.training.modules.common import interface_weights, normalize_forecast_batch, per_field_mae, pooled_l1_loss
from nucleus.training.moe import moe_metrics, router_loss, update_router_bias
from nucleus.training.push_forward import push_forward, sample_num_unrolls


@dataclass(frozen=True)
class Nucleus2MoEDivFreeProjectionTrainingSettings:
    target_timesteps: int
    load_balance_loss_weight: float = 1e-5
    z_loss_weight: float = 1e-5
    flip_probability: float = 0.5
    min_noise_std: float = 1e-3
    max_noise_std: float = 5e-2
    # unrolls past the first target window; the dataset's unroll_steps minus one
    max_unrolls: int = 0
    push_forward_probability: float = 0.0
    # the training loss weight of the cells within interface_loss_band_cells of the
    # target's interface, against 1 for the rest; 1 weighs every cell alike
    interface_loss_weight: float = 1.0
    interface_loss_band_cells: float = 3.0
    # without it, the divergence source learns only from the velocity loss downstream of it
    supervise_div_source: bool = False
    # score the divergence-free parts of the face velocities rather than the full
    # velocities, so the loss leaves the dilatational part to the divergence source
    solenoidal_velocity_target: bool = True


def target_div_source(target: FlashXBatch, grid: CellCenteredGrid) -> torch.Tensor:
    """The divergence of the target velocity, which supervises the divergence source.
    The velocities are normalized by a pure scale, so the divergence of the
    normalized velocity is the normalized physical one."""
    return divergence_centers_from_faces(target.fields["velfacex"], target.fields["velfacey"], grid.dx, grid.dy)


def solenoidal_faces(batch: FlashXBatch, grid: CellCenteredGrid) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (B, T, H, W + 1) and (B, T, H + 1, W) divergence-free part of the batch's face
    velocity: the velocity minus grad(phi), where laplacian(phi) is its divergence.
    The curl model's dilatational velocity is a grad(phi) with the same boundary
    conditions, so this removes it and leaves the curl of the stream function."""
    return div_projection(batch.fields["velfacex"], batch.fields["velfacey"], grid.dx, grid.dy)


class Nucleus2MoEDivFreeProjectionTraining(nn.Module):
    """Trains a Nucleus2MoEDivFreeProjection on raw FlashXForecastBatches: flips the raw batch,
    normalizes it, adds noise to the normalized input, optionally pushes the input
    forward through the model, then takes the L1 loss over the sdf, the temperature,
    the face velocities (only their divergence-free parts with solenoidal_velocity_target)
    and, with supervise_div_source, the divergence source, weighted toward the
    interface by interface_loss_weight, and the router loss. The predicted face
    velocities are the network's own unless the model's config sets use_div_projection."""

    def __init__(self, model: Nucleus2MoEDivFreeProjection, settings: Nucleus2MoEDivFreeProjectionTrainingSettings):
        super().__init__()
        self.model = model
        self.settings = settings
        # flips negate velocities and mirror the heater about the domain bounds,
        # which only holds for raw values
        self.raw_augmentation = ForecastAugmentation([FlipAugmentation(settings.flip_probability)])
        self.normalized_augmentation = ForecastAugmentation([
            GaussianNoiseAugmentation(settings.min_noise_std, settings.max_noise_std),
        ])

    def training_step(self, batch: FlashXForecastBatch, step: int) -> StepOutput:
        batch = self.normalized_augmentation(normalize_forecast_batch(self.model.normalizer, self.raw_augmentation(batch)))
        num_unrolls = sample_num_unrolls(self.settings.max_unrolls, self.settings.push_forward_probability)
        batch = push_forward(self._next_window, batch, num_unrolls, self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, batch.target)
        weights = interface_weights(
            self.model.normalizer,
            batch.target,
            self.settings.interface_loss_weight,
            self.settings.interface_loss_band_cells,
        )
        data_loss = pooled_l1_loss(pairs, weights)
        loss = data_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {"train/loss": loss, "train/data_loss": data_loss, "train/num_unrolls": num_unrolls}
        metrics |= per_field_mae(pairs, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        target = batch.target.head_time_window(self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, target)
        data_loss = pooled_l1_loss(pairs)
        metrics = {"val/loss": data_loss, "val/data_loss": data_loss}
        return metrics | per_field_mae(pairs, "val") | moe_metrics(output.moe_outputs, "val")

    @torch.no_grad()
    def gradient_metrics(self) -> Metrics:
        head = self.model.debed
        grads = torch.cat([param.grad.flatten() for param in head.parameters() if param.grad is not None])
        weights = torch.cat([param.flatten() for param in head.parameters() if param.ndim >= 2])
        return {
            "train/debed/grad_norm": grads.norm(),
            "train/debed/grad_absmax": grads.abs().max(),
            "train/debed/weight_std": weights.std(),
            "train/debed/weight_absmax": weights.abs().max(),
        }

    def _next_window(self, window: FlashXBatch) -> FlashXBatch:
        return self.model(window).prediction.tail_time_window(self.settings.target_timesteps)

    def _prediction_target_pairs(
        self, output: Nucleus2MoEDivFreeProjectionOutput, target: FlashXBatch
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        prediction = output.prediction.tail_time_window(target.num_timesteps)
        window = slice(-target.num_timesteps, None)
        pairs = {name: (prediction.fields[name], target.fields[name]) for name in self.model.field_names}
        if self.settings.solenoidal_velocity_target:
            del pairs["velfacex"], pairs["velfacey"]
            predicted_velx, predicted_vely = solenoidal_faces(prediction, output.grid)
            target_velx, target_vely = solenoidal_faces(target, output.grid)
            pairs["solenoidal_velfacex"] = (predicted_velx, target_velx)
            pairs["solenoidal_velfacey"] = (predicted_vely, target_vely)
        if self.settings.supervise_div_source:
            pairs["div_source"] = (output.div_source[:, window], target_div_source(target, output.grid))
        return pairs
