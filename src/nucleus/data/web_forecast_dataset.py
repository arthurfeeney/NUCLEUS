from typing import List, Optional
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset
import webdataset as wds

from nucleus.data.batching import make_data, Data
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import Normalizer


class WebForecastDataset(IterableDataset):
    """
    Streams training samples from webdataset shards produced by hdf5_dataset_to_webdataset.py.

    Each shard sample contains:
      - .npz : float32 array of shape [T, H, W, 4] (dfun, temperature, velx, vely)
      - .json : simulation parameters dict

    Only the first `history_time_window + future_time_window` timesteps of each
    chunk are used. The rest are discarded.
    """

    def __init__(
        self,
        shard_urls: List[str],
        history_time_window: int,
        future_time_window: int,
        fluid_params: List[str],
        heater_params: List[str],
        global_params: List[str],
        layout: str,
        normalizer: Optional[Normalizer],
        augment: bool,
        shuffle_buffer: int = 1000,
    ):
        super().__init__()
        self.shard_urls = shard_urls
        self.history_time_window = history_time_window
        self.future_time_window = future_time_window
        self.fluid_params = fluid_params
        self.heater_params = heater_params
        self.global_params = global_params
        self.layout = layout
        self.normalizer = normalizer
        self.augment = augment
        self.shuffle_buffer = shuffle_buffer

    def _decode_sample(self, sample: dict) -> Data:
        fields = torch.from_numpy(sample["npz"]["fields"].copy())  # [T, H, W, 4]
        sim_params = sample["json"]

        inp = fields[:self.history_time_window]
        tgt = fields[self.history_time_window:self.history_time_window + self.future_time_window]

        bulk_temp = int(sim_params["bulk_temp"])
        if self.normalizer is not None:
            inp = self.normalizer.normalize(inp, bulk_temp)
            tgt = self.normalizer.normalize(tgt, bulk_temp)
            sim_params = self.normalizer.normalize_params([sim_params])[0]

        if self.augment and random.random() < 0.5:
            inp = torch.flip(inp, dims=[2])
            tgt = torch.flip(tgt, dims=[2])

        inp = convert_layout(inp, self.layout)
        tgt = convert_layout(tgt, self.layout)

        return make_data(
            input=inp.float(),
            target=tgt.float(),
            fluid_params_dict=sim_params,
            downsample_factor=1,
        )

    def __iter__(self):
        pipeline = (
            wds.WebDataset(self.shard_urls, shardshuffle=self.augment)
            .decode()
        )
        if self.augment:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        pipeline = pipeline.map(self._decode_sample)
        yield from pipeline
