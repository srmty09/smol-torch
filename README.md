# Smol-PyTorch

A minimal PyTorch implementation in Julia, designed to understand the core concepts of deep learning frameworks.

## Project Overview

This project aims to replicate PyTorch's core functionality in Julia, focusing on:
- Tensor operations and automatic differentiation
- Neural network layers and modules
- Training utilities (loss functions, optimizers)
- Basic examples and demonstrations

## Current Implementation Status

### Core Components

#### Tensor Implementation
- **Status**: Implemented
- **File**: `core/tensor.jl`
- **Features**:
  - Basic Tensor struct with data, shape, and gradient storage
  - Support for Float64 and Int64 data types
  - Memory layout management (row-major to column-major conversion)
  - Stride calculation for efficient indexing
  - Gradient tracking capability

#### Operations
- **Status**: Implemented
- **File**: `core/ops.jl`
- **Features**:
  - Element-wise operations: `+`, `-`, `*`, `/`, `//`
  - Mathematical functions: `exp`, `log`, `abs`, `sqrt`
  - Activation functions: `sigmoid`, `tanh`, `relu`
  - Matrix operations: `dot` (matrix multiplication)
  - Reduction operations: `sum`, `mean`
  - Shape manipulation: `reshape_`

### Missing Components

#### Neural Network Module
- **Status**: Not implemented
- **Required Files**:
  - `nn/module.jl` - Abstract Module type
  - `nn/linear.jl` - Linear layer implementation
  - `nn/activation.jl` - Activation function layers
  - `nn/init.jl` - Module initialization

#### Training Components
- **Status**: Not implemented
- **Required Files**:
  - `train/loss.jl` - Loss functions (MSE, CrossEntropy)
  - `train/optimizer.jl` - Optimizers (SGD, Adam)
  - `train/init.jl` - Training utilities

#### Automatic Differentiation
- **Status**: Not implemented
- **Required Files**:
  - `core/grad.jl` - Backpropagation engine
  - `core/utils.jl` - Broadcasting and shape utilities

#### Examples and Utilities
- **Status**: Not implemented
- **Required Files**:
  - `examples/xor.jl` - XOR classification example
  - `examples/mnist.jl` - MNIST digit classification
  - `utils/data_loader.jl` - Data loading utilities
  - `utils/show.jl` - Tensor display utilities

## Project Structure

```
smol-pytorch/
├── README.md                   # Project documentation
├── filestruct.md              # Planned file structure
├── test.jl                    # Basic testing script
│
├── core/                      # Core components
│   ├── tensor.jl              # Tensor implementation
│   ├── ops.jl                 # Mathematical operations
│   ├── grad.jl                # Backpropagation engine (TODO)
│   ├── utils.jl               # Utility functions (TODO)
│   └── init.jl                # Core initialization (TODO)
│
├── nn/                        # Neural network components (TODO)
│   ├── module.jl              # Abstract Module type
│   ├── linear.jl              # Linear layer
│   ├── activation.jl          # Activation functions
│   └── init.jl                # NN initialization
│
├── train/                     # Training components (TODO)
│   ├── loss.jl                # Loss functions
│   ├── optimizer.jl           # Optimizers
│   └── init.jl                # Training initialization
│
├── examples/                  # Example scripts (TODO)
│   ├── xor.jl                 # XOR classification
│   └── mnist.jl               # MNIST classification
│
└── utils/                     # Utilities (TODO)
    ├── data_loader.jl         # Data loading
    └── show.jl                # Display utilities
```

## Usage Examples

### Basic Tensor Operations

```julia
include("core/ops.jl")
include("core/tensor.jl")

# Create tensors
arr = [-1.0 2.0 3.0; 4.0 -5.0 6.0]
a = Tensor(arr, true)

# Apply operations
a = relu(a)
result = exp(a)
println(result)
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (In Progress)
- [x] Basic Tensor implementation
- [x] Element-wise operations
- [x] Mathematical functions
- [ ] Automatic differentiation engine
- [ ] Broadcasting utilities

### Phase 2: Neural Network Components
- [ ] Abstract Module type
- [ ] Linear layer implementation
- [ ] Activation function layers
- [ ] Module parameter management

### Phase 3: Training Framework
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Optimizers (SGD, Adam)
- [ ] Training loops
- [ ] Gradient computation

### Phase 4: Examples and Utilities
- [ ] XOR classification example
- [ ] MNIST digit classification
- [ ] Data loading utilities
- [ ] Tensor visualization

### Phase 5: Advanced Features
- [ ] Convolutional layers
- [ ] Recurrent layers
- [ ] Batch normalization
- [ ] Dropout layers

## Key Implementation Details

### Tensor Design
- Uses row-major to column-major conversion for efficient matrix operations
- Supports both gradient tracking and non-tracking modes
- Implements stride-based indexing for memory efficiency

### Operation System
- Overloads Julia's Base operators for intuitive syntax
- Implements element-wise operations with shape validation
- Provides mathematical functions with gradient preservation

### Memory Management
- Efficient memory layout for matrix operations
- Stride calculation for optimal indexing
- Copy-on-write semantics for data safety

## Testing

Run the basic test script:
```bash
julia test.jl
```

## Contributing

This is an educational project. Contributions are welcome for:
- Implementing missing components
- Improving existing implementations
- Adding new features
- Documentation improvements

## License

This project is for educational purposes.

## References

- PyTorch documentation and source code
- Julia programming language documentation
- Deep learning fundamentals and automatic differentiation 