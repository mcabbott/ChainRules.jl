#####
##### `sum`
#####

function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

function rrule(::typeof(sum), x::AbstractArray{T}; dims=:) where {T<:Number}
    y = sum(x; dims=dims)
    function sum_pullback(ȳ)
        # broadcasting the two works out the size no-matter `dims`
        x̄ = InplaceableThunk(
            @thunk(broadcast((_,y1)->y1, x, ȳ)), # last∘tuple
            x -> x .+= x̄
        )
        return (NO_FIELDS, x̄)
    end
    return y, sum_pullback
end

function frule(
    (_, _, ẋ),
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    y = sum(abs2, x; dims=dims)
    ∂y = if dims isa Colon
        2 * real(dot(x, ẋ))
    elseif VERSION ≥ v"1.2" # multi-iterator mapreduce introduced in v1.2
        mapreduce(+, x, ẋ; dims=dims) do xi, dxi
            2 * _realconjtimes(xi, dxi)
        end
    else
        2 * sum(_realconjtimes.(x, ẋ); dims=dims)
    end
    return y, ∂y
end

function rrule(
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    y = sum(abs2, x; dims=dims)
    function sum_abs2_pullback(ȳ)
        x_thunk = InplaceableThunk(
            @thunk(2 .* real.(ȳ) .* x),
            dx -> dx .+= 2 .* real.(ȳ) .* x
        )
        return (NO_FIELDS, DoesNotExist(), x_thunk)
    end
    return y, sum_abs2_pullback
end

#####
##### `cumsum`
#####

function frule((_, ẋ), ::typeof(cumsum), x; dims=1)
    return cumsum(x; dims=dims), cumsum(ẋ; dims=dims)
end

function rrule(::typeof(cumsum), x::AbstractArray{T}; dims=1) where {T<:Number}
    y = cumsum(x; dims=dims)
    function cumsum_pullback(ȳ)
        x_thunk = @thunk(reverse(cumsum(reverse(ȳ; dims=dims); dims=dims); dims=dims))
        return (NO_FIELDS, x_thunk)
    end
    # function cumsum_pullback(ȳ::StridedArray)
    #     # x_thunk = @thunk(reverse!(cumsum!(reverse(ȳ; dims=dims); dims=dims); dims=dims))
    #     x_thunk = @thunk begin
    #         tmp = reverse(ȳ; dims=dims)
    #         cumsum!(tmp, tmp; dims=dims)
    #         reverse!(tmp; dims=dims) # fails
    #     end
    #     return (NO_FIELDS, x_thunk)
    # end
    return y, cumsum_pullback
end

