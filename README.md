# NN Engine & Backpropagation

A scalar-valued autograd engine with backpropagation implemented from scratch in
NumPy, in the style of Karpathy's micrograd. It defines a `Value` class that
tracks a computation graph and computes gradients via reverse-mode automatic
differentiation — then builds a small neural network (Neuron / Layer / MLP) on top of
it and trains it on a toy dataset.

## What's inside

- `engine.ipynb` — the full walkthrough:
  - `Value` class: scalar values with `data`, `grad`, and `_backward()` for each op
    (`+`, `*`, `tanh`, `exp`, and more)
  - Topological-sort based `backward()` pass over the computation graph
  - Graphviz visualization of the graph (`draw_dot`)
  - `Neuron`, `Layer`, and `MLP` classes built on `Value`
  - Training loop: forward pass, loss, zeroing grads, gradient descent updates
- `requirements.txt` — Python dependencies
- A bundled Graphviz Windows installer (plus its extracted folder) — needed only
  for rendering the computation-graph diagrams on Windows

## Running it

```bash
pip install -r requirements.txt
jupyter notebook engine.ipynb
```

For the graph diagrams you also need the Graphviz system package installed; the
core autograd engine itself only needs NumPy and matplotlib.
