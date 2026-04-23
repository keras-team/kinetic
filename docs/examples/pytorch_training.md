# PyTorch Training

Kinetic can run PyTorch workloads on cloud GPUs. Since Kinetic executes arbitrary Python functions remotely, any PyTorch code that runs locally will run the same way on a provisioned GPU node.

## Setup

Add `torch` to your project's `requirements.txt`:

```text
torch
torchvision
```

Kinetic will install these in the remote container automatically. See [Managing Dependencies](../guides/dependencies.md) for details on how dependency detection works.

## Basic Usage

```python
import kinetic

@kinetic.run(accelerator="gpu-l4")
def train():
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Simple feedforward network
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Dummy data
    x = torch.randn(512, 10, device=device)
    y = torch.randn(512, 1, device=device)

    for epoch in range(20):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}")

    return loss.item()

final_loss = train()
```

## Multi-GPU Training

For nodes with multiple GPUs, use `torch.nn.DataParallel` to split batches across devices.

```python
import kinetic

@kinetic.run(accelerator="gpu-a100x4")
def train_multi_gpu():
    import torch
    import torch.nn as nn

    device = torch.device("cuda")
    print(f"GPUs available: {torch.cuda.device_count()}")

    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(2048, 10, device=device)
    y = torch.randn(2048, 1, device=device)

    for epoch in range(20):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
```

## GPU Selection

See [Accelerator Support](../accelerators.md) for the full list of GPUs, multi-GPU counts, and TPU configurations.

Use `spot=True` to reduce costs for fault-tolerant workloads:

```python
@kinetic.run(accelerator="gpu-a100", spot=True)
def train():
    ...
```

## Related pages

- [Dependencies](../guides/dependencies.md) — how `torch` gets installed in
  the remote container.
- [Accelerators](../accelerators.md) — full list of GPUs and
  multi-GPU configurations.
- [Cost Optimization](../guides/cost_optimization.md) — spot capacity for
  GPU workloads.
