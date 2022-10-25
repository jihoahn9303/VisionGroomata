import time

import torch

torch.set_printoptions(precision=2, sci_mode=False)
# torch.manual_seed(42)

SHAPE = 100
STEP_SIZE = 0.001


def loss_fn(parameter: torch.Tensor) -> torch.Tensor:
    """simulates expensive calculations."""
    print("Calculating....")
    time.sleep(1)
    return (
        parameter + parameter.cos() + parameter.pow(2) + (-parameter).pow(2).exp()
    ).sum()


def evaluate_delta(parameter: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (loss_fn(parameter + STEP_SIZE * delta) - loss_fn(parameter)) / STEP_SIZE


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x, y)


parameter = torch.ones(SHAPE)
parameter.requires_grad_(True)

loss = loss_fn(parameter)
loss.backward()

gradient: torch.Tensor = parameter.grad

# # search best delta that can mininize loss
# for i in range(1000):
#     delta = F.normalize(torch.randn(SHAPE), dim=0)  # fix norm -> search direction
#     # print(evaluate_delta(parameter, delta))
#     print(inner_product(gradient, delta))


# delta = F.normalize(-gradient, dim=0)
# print(inner_product(gradient, delta))

# in-place operation을 위하여, 파라미터 추적을 방지
with torch.no_grad():
    parameter -= STEP_SIZE * gradient
