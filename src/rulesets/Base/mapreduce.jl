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

#####
##### `prod`
#####

function frule((_, ẋ), ::typeof(prod), x; dims=:)
    # ???
    return prod(x; dims=dims), sum(ẋ; dims=dims)
end

function rrule(::typeof(prod), x::AbstractArray{T}; dims=:) where {T<:Number}
    y = prod(x; dims=dims)
    function prod_pullback(dy)
        x_thunk = if dims == (:)
            InplaceableThunk(
                @thunk(∇prod(x, dy, y)),  # This is usually y ./ x .* dy
                dx -> ∇prod!(dx, x, dy, y)
            )
        elseif any(iszero, x)  # Only cases where ./x would give NaN
            InplaceableThunk(
                @thunk(∇prod_dims(dims, x, dy, y)),
                dx -> ∇prod_dims!(dx, dims, x, dy, y)
            )
        else
            InplaceableThunk(
                @thunk(y ./ x .* dy),
                dx -> dx .+= y ./ x .* dy
            )
        end
        (NO_FIELDS, x_thunk)
    end
    return y, prod_pullback
end

function ∇prod_dims(dims, x, dy=fill!(sum(x; dims=dims), 1), y=prod(x; dims=dims))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T), 0)
    ∇prod_dims!(dx, dims, x, dy, y)
    dx
end

function ∇prod_dims!(dx, dims, x, dy, y)
    iters = ntuple(d -> d in dims ? tuple(:) : axes(x,d), ndims(x))
    # @show x y dx dy
    for ind in Iterators.product(iters...)
        # if y isa AbstractArray
            jay = map(i -> i isa Colon ? 1 : i, ind)
            # @show ind jay
            @views ∇prod!(dx[ind...], x[ind...], dy[jay...], y[jay...])
        # else
        #     @show ind
        #     @views ∇prod!(dx[ind...], x[ind...], dy, y)
        # end
    end
    dx
end


# To opt out of this mapslices thing, and accept NaN instead, you could define:
# ∇prod_dims!(dx, dims, x::CuArray, dy, y) = dx .+= y ./ x .* dy

function ∇prod(x, dy::Number=1, y::Number=prod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = similar(x, T) .= 0
    ∇prod!(dx, x, y, dy)
    dx
end

function ∇prod!(dx, x, dy::Number=1, y::Number=prod(x))
    numzero = count(iszero, x)
    if numzero == 0  # This can happen while y==0, if there are several small xs
        dx .+= y ./ x .* dy
    elseif numzero > 1
        dx
    else
        ∇prod_one_zero!(dx, x, dy)
    end
    dx
end

function ∇prod_one_zero!(dx, x, dy::Number=1)  # Assumes exactly one x is zero
    i_zero = 0
    p_rest = one(promote_type(eltype(x), typeof(dy)))
    for i in eachindex(x)
        xi = @inbounds x[i]
        p_rest *= ifelse(iszero(xi), one(eltype(x)), xi)
        i_zero = ifelse(iszero(xi), i, i_zero)
    end
    dx[i_zero] += p_rest * dy
    dx
end



