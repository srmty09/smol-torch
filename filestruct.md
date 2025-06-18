PyTorchClone.jl/
├── main.jl                     # Entry point: runs demos/tests
├── README.md                   # Project documentation
│
├── core/                       # Core components: Tensor, ops, autograd
│   ├── tensor.jl               # Tensor struct: data, grad, shape, etc.
│   ├── ops.jl                  # Manual math ops: +, *, matmul, etc.
│   ├── grad.jl                 # Backprop engine (dynamic graph)
│   ├── utils.jl                # Helpers: broadcasting, shape checks
│   └── init.jl                 # include(...) all core files
│
├── nn/                         # Neural network components
│   ├── module.jl               # Abstract Module type
│   ├── linear.jl               # Linear layer
│   ├── activation.jl           # ReLU, Sigmoid, etc.
│   └── init.jl                 # include(...) all nn files
│
├── train/                      # Training components
│   ├── loss.jl                 # Loss functions (MSE, CrossEntropy)
│   ├── optimizer.jl            # Optimizers (SGD, Adam)
│   └── init.jl                 # include(...) all training files
│
├── examples/                   # Simple training/testing scripts
│   ├── xor.jl                  # XOR toy example
│   └── mnist.jl                # MNIST demo (optional custom loader)
│
└── utils/                      # Misc helpers
    ├── data_loader.jl          # Data loading (if any)
    └── show.jl                 # Custom display for Tensors

