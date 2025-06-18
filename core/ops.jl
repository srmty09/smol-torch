include("tensor.jl")
import Base: +, -, *, /, //
import Base: exp, log, abs, sqrt


# Element-wise addition
function +(a::Tensor, b::Tensor)::Tensor
    @assert a.size_ == b.size_ "Size mismatch in addition"
    res_data = Float64[]
    for i in 1:a.size_
        push!(res_data, a.data_[i] + b.data_[i])
    end
    return Tensor(res_data, true)
end

# Element-wise subtraction
function -(a::Tensor, b::Tensor)::Tensor
    @assert a.size_ == b.size_ "Size mismatch in subtraction"
    res_data = Float64[]
    for i in 1:a.size_
        push!(res_data, a.data_[i] - b.data_[i])
    end
    return Tensor(res_data, true)
end

# Element-wise multiplication
function *(a::Tensor, b::Tensor)::Tensor
    @assert a.size_ == b.size_ "Size mismatch in multiplication"
    res_data = Float64[]
    for i in 1:a.size_
        push!(res_data, a.data_[i] * b.data_[i])
    end
    return Tensor(res_data, true)
end

# Element-wise division
function /(a::Tensor, b::Tensor)::Tensor
    @assert a.size_ == b.size_ "Size mismatch in division"
    res_data = Float64[]
    for i in 1:a.size_
        if b.data_[i] == 0
            error("Division by zero at index $i")
        end
        push!(res_data, a.data_[i] / b.data_[i])
    end
    return Tensor(res_data, true)
end

# Element-wise integer division
function //(a::Tensor, b::Tensor)::Tensor
    @assert a.size_ == b.size_ "Size mismatch in integer division"
    res_data = Int64[]
    for i in 1:a.size_
        if b.data_[i] == 0
            error("Division by zero at index $i")
        end
        push!(res_data, Int64(a.data_[i] ÷ b.data_[i]))
    end
    return Tensor(res_data, true)
end

function dot(a::Tensor, b::Tensor)::Tensor
    @assert a.shape_ !== nothing && b.shape_ !== nothing "Shapes must be defined"
    @assert length(a.shape_) == 2 && length(b.shape_) == 2 "Only 2D dot supported"
    @assert a.shape_[2] == b.shape_[1] "Incompatible shapes for matrix multiplication"

    m, n = a.shape_[1], a.shape_[2]
    _, p = b.shape_[1], b.shape_[2]


    A = a.data_
    B = b.without_transpose

    @assert A !== nothing && B !== nothing "Missing data for matmul"

    result_col_major = zeros(Float64, m * p)

    for i in 0:m-1
        for j in 0:p-1
            sum = 0.0
            for k in 0:n-1
                a_idx = i * n + k         
                b_idx = k + j * n         
                sum += A[a_idx + 1] * B[b_idx + 1]
            end
            result_col_major[i + j * m + 1] = sum
        end
    end


    result_matrix = reshape(result_col_major, m, p)
    result_row_major = reshape(copy(transpose(result_matrix)), m * p)

    return Tensor(result_row_major, false)
end

function sum(a::Tensor)::Float64
    sum_ = 0.0
    for i in 1:a.size_
        sum_ += a.data_[i]
    end    
    return sum_
end

function mean(a::Tensor)::Float64
    sum_ = sum(a)
    return sum_ / Float64(a.size_)
end

function reshape_(a::Tensor, shape::Tuple)::Nothing
    a.shape_ = collect(shape)
    a.stride_ = calculate_stride(shape)
    return nothing
end

function exp(a::Tensor)::Tensor
    res_data = [exp(x) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end

function log(a::Tensor)::Tensor
    res_data = [log(x) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end

function sigmoid(a::Tensor)::Tensor
    res_data = [1.0 / (1.0 + exp(-x)) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end

function tanh(a::Tensor)::Tensor
    res_data = [tanh(x) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end

function abs(a::Tensor)::Tensor
    res_data = [abs(x) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end

function sqrt(a::Tensor)::Tensor
    res_data = [sqrt(x) for x in a.data_]
    return Tensor(res_data, a.need_grad)
end


function relu(a::Tensor)::Tensor
    for i in 1:a.size_
        if a.data_[i]<0
            a.data_[i]=0
        else
            continue
        end
    end
    return a 
end

