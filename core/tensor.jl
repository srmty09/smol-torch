mutable struct Tensor
    data_::Union{Nothing, Array{Int64,1}, Array{Float64,1}}
    shape_::Union{Nothing, Vector{Int}}
    size_::Union{Nothing, Int64}
    need_grad::Bool
    grad_::Union{Nothing, Array{Float64,1}}
    stride_::Union{Nothing, Vector{Int}}
    ops_::Union{Nothing, String}
    without_transpose::Union{Nothing,Array{Int64,1},Array{Float64,1}}

    # default constructor
    function Tensor()
        println("Give data to Tensor!")
        return new(nothing, nothing, nothing, false, nothing, nothing, nothing)
    end



    # main constructor for Float64 data
    function Tensor(data::Array{Float64}, need_grad::Bool)
        shape_of_data = size(data)
        length_arr = prod(shape_of_data)
        data1=copy(data)
        if ndims(data) > 1
            data1=reshape(data1,length_arr)
            data = copy(transpose(data))   
            data = reshape(data, length_arr)
        end

        stride = calculate_stride(shape_of_data)

        if need_grad
            println("tensor created successfully")
            return new(data, collect(shape_of_data), length_arr, true, zeros(length_arr), stride, nothing,data1)
        else
            println("tensor created successfully")
            return new(data, collect(shape_of_data), length_arr, false, nothing, stride, nothing,data1)
        end
    end

    # main constructor for Int64 data
    function Tensor(data::Array{Int64}, need_grad::Bool)
        shape_of_data = size(data)
        length_arr = prod(shape_of_data)
        data1=copy(data)
        if ndims(data) > 1
            data1=reshape(data1,length_arr)
            data = copy(transpose(data)) 
            data = reshape(data, length_arr)
        end

        stride = calculate_stride(shape_of_data)

        if need_grad
            println("tensor created successfully")
            return new(data, collect(shape_of_data), length_arr, true, zeros(length_arr), stride, nothing,data1)
        else
            println("tensor created successfully")
            return new(data, collect(shape_of_data), length_arr, false, nothing, stride, nothing,data1)
        end
    end
end


# helper to calculate stride
function calculate_stride(a::Tuple)::Vector{Int64}
    stride_arr = []
    element = 1
    push!(stride_arr, element)
    for i in length(a):-1:2
        element *= a[i]
        push!(stride_arr, element)
    end
    reverse!(stride_arr)
    return stride_arr
end