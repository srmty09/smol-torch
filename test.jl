include("core/ops.jl")
include("core/tensor.jl")


arr = [-1.0 2.0 3.0;
       4.0 -5.0 6.0]  

arr1 = [1.0 2.0 3.0;
        3.0 4.0 4.0;
        5.0 6.0 5.0]    

a=Tensor(arr,true)
b=Tensor(arr1,true)

a=relu(a)
c=exp(a)
println(c)