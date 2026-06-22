import argparse
import torch
from triton.testing import do_bench

from nucleus.models import Nucleus2MoE, Nucleus2MoEConfig
from nucleus.data.batching import CollatedBatch

parser = argparse.ArgumentParser()
parser.add_argument("--patch_size", type=int, default=32)
parser.add_argument("--embed_dim", type=int, default=512)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--moe_intermediate_dim", type=int, default=512)
parser.add_argument("--backward", action="store_true", default=False)
args = parser.parse_args()

config = Nucleus2MoEConfig(
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    processor_blocks=12,
    num_experts=6,
    topk=2,
    moe_intermediate_dim=args.moe_intermediate_dim,
    embed_dtype=torch.bfloat16,
    debed_dtype=torch.bfloat16,
    activation_dtype=torch.torch.bfloat16,
    attention_dtype=torch.bfloat16,
    moe_dtype=torch.bfloat16
)

model = Nucleus2MoE(config).to("cuda")
input = CollatedBatch(
    input=torch.randn((8, 8, 512, 512, 4), dtype=torch.float32, device="cuda"),
    target=None,
    sim_params_dict={},
    sim_params_tensor=torch.randn((8, 14), dtype=torch.float32, device="cuda"),
    dx=torch.tensor(8, dtype=torch.float32, device="cuda"),
    dy=torch.tensor(8, dtype=torch.float32, device="cuda"),
    x_grid=None,
    y_grid=None
)

if args.backward:
    def forward_and_backward():
        output, _ = model(input)
        output.sum().backward()
    ms, min_ms, max_ms = do_bench(lambda: forward_and_backward(), quantiles=[0.5, 0.2, 0.8])
else:
    ms, min_ms, max_ms = do_bench(lambda: model(input), quantiles=[0.5, 0.2, 0.8])

print(torch.cuda.get_device_name(input.input.device), torch.cuda.get_device_capability(input.input.device))
print(f"{ms} ms")
print(f"{min_ms} ms")
print(f"{max_ms} ms")