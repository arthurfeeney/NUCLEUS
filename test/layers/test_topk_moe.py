import torch

from nucleus.layers.moe.topk_moe import TopkRouterWithBias


def test_the_router_bias_rises_for_underloaded_experts_and_falls_for_the_rest():
    router = TopkRouterWithBias(4, hidden_dim=8, topk=1, bias_update_rate=0.1, softmax_first=False)
    # 12 tokens, so the target load is 3 per expert
    router.update_router_bias(torch.tensor([1, 2, 5, 4]))
    torch.testing.assert_close(router.router_bias, torch.tensor([0.1, 0.1, -0.1, -0.1]))
    router.update_router_bias(torch.tensor([1, 2, 5, 4]))
    torch.testing.assert_close(router.router_bias, torch.tensor([0.2, 0.2, -0.2, -0.2]))
