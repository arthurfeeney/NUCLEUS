from dataclasses import dataclass
from typing import List
import torch
from bubbleformer.data.batching import Data
from bubbleformer.data import BubbleForecast
from bubbleformer.layers.moe.topk_moe import TopkMoEOutput
from bubbleformer.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from bubbleformer.utils.sdf_reinit import sdf_reinit

@dataclass
class TestResults:
    preds: torch.Tensor
    targets: torch.Tensor
    p: PhysicalMetrics
    b: BubbleMetrics
    moe_outputs: List[TopkMoEOutput]
    fluid_params: dict

def run_test(model, test_file_path: str, max_timesteps: int):
    downsample_factor = 8
    test_dataset = BubbleForecast(
        filenames=[test_file_path],
        input_fields=["dfun", "temperature", "velx", "vely"],
        output_fields=["dfun", "temperature", "velx", "vely"],
        norm="none",    
        downsample_factor=downsample_factor,
        time_window=5,
        start_time=200,
        return_fluid_params=True,
    )

    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    preds = []
    targets = []
    timesteps = []
    moe_outputs = []

    with torch.inference_mode():
        for itr in range(0, max_timesteps, skip_itrs):
            print(f"Processing timestep {itr} / {max_timesteps}")
            
            data: Data = test_dataset[itr]  
            inp = data.input
            tgt = data.target
            fluid_params = data.fluid_params_tensor
            if len(preds) > 0:
                inp = preds[-1]

            inp = inp.cuda().to(torch.float32).unsqueeze(0)
            fluid_params = fluid_params.cuda().to(torch.float32).unsqueeze(0)
            
            pred, moe_output = model(inp, fluid_params)
            
            # NOTE: only tracking moe outputs for the first layer
            # all tensor are moved to the CPU
            moe_outputs.append(moe_output[0].detach().to('cpu'))

            # clip pred temperature to valid range, between liquid bulk temp and heater temp.
            pred[:, :, 1] = torch.clamp(
                pred[:, :, 1], 
                min=data.fluid_params_dict["bulk_temp"], 
                max=data.fluid_params_dict["heater"]["wallTemp"]
            )
            
            # Reinitialize the SDF at each timestep, batch dim is 1
            #pred[0, :, 0] = sdf_reinit(pred[0, :, 0], dx=1/32 * downsample_factor)

            pred = pred.to(torch.float32).squeeze(0)
            pred = pred.detach().cpu()
            tgt = tgt.detach().cpu()

            preds.append(pred)
            targets.append(tgt)
            timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))

    preds = torch.cat(preds, dim=0)[None, ...]         # 1, T, C, H, W
    targets = torch.cat(targets, dim=0)[None, ...]     # 1, T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T,

    topk_indices = [moe_output.topk_indices.squeeze(0) for moe_output in moe_outputs]
    topk_indices = torch.cat(topk_indices, dim=0) # (T, H, W, topk)

    print("-"*100)
    print(f"Rollout Statistics on {test_file_path}:")
    dx = 1/32 * downsample_factor
    dy = dx
    p = physical_metrics(
        preds[:, :, 0], 
        preds[:, :, 1], 
        preds[:, :, 2], 
        preds[:, :, 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        # TODO: get from dataset
        xcoords=torch.arange(-8, 8, dx) + dx / 2,
        # TODO: get from dataset
        dx=dx, 
        dy=dy
    )
    b = bubble_metrics(preds[:, :, 0], preds[:, :, 2], preds[:, :, 3], dx=dx, dy=dy)
    
    # NOTE: the test dataset is only one file, so we can take the first index in fluid_params
    return TestResults(preds, targets, p, b, moe_outputs, fluid_params=test_dataset.fluid_params[0])