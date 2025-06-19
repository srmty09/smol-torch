using Statistics

mutable struct LinearRegression
    x_data::Union{Nothing, Array{Float64,2}}
    y_data::Union{Nothing, Array{Float64,2}}
    weights::Union{Nothing, Array{Float64,2}}
    learning_rate::Float64
    epochs::Int64

    function LinearRegression()
        return new(nothing, nothing, nothing, 0.012, 10)
    end
end

function set_epochs(model::LinearRegression, num::Int64)
    model.epochs = num
end

function set_lr(model::LinearRegression, num::Float64)
    model.learning_rate = num
end

function weight(num_features::Int)
    return reshape(randn(num_features), (num_features, 1))
end

function fit(model::LinearRegression, x_train::Array{Float64,2}, y_train::Array{Float64,1})
    n = size(x_train, 1)
    x_stacked = hcat(ones(n), x_train)  

    model.x_data = x_stacked
    model.y_data = reshape(y_train, :, 1)  

    num_features = size(x_stacked, 2)
    model.weights = weight(num_features)

    for i in 1:model.epochs
        preds = model.x_data * model.weights

        error = preds - model.y_data
        print("Epoch ", i)
        print(" | Loss: ")
        println(mean(error .^ 2))
        gradient = (1 / n) * (model.x_data' * error)
        model.weights .-= model.learning_rate * gradient
    end
end

function inline_predict(model::LinearRegression, x::Array{Float64,2})
    n = size(x, 1)
    x_stacked = hcat(ones(n), x)
    return x_stacked * model.weights
end

using Statistics

"Mean Squared Error"
function mse(y_true::Array{Float64,2}, y_pred::Array{Float64,2})
    return mean((y_true .- y_pred).^2)
end

"Root Mean Squared Error"
function rmse(y_true::Array{Float64,2}, y_pred::Array{Float64,2})
    return sqrt(mse(y_true, y_pred))
end

"R-squared Score"
function r2_score(y_true::Array{Float64,2}, y_pred::Array{Float64,2})
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end



