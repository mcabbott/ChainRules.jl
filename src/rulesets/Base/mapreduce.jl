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
            @thunk(broadcast(last∘tuple, x, ȳ)),
            x -> x .+= ȳ
        )
        return (NoTangent(), x̄)
    end
    return y, sum_pullback
end

# Can't map over Adjoint/Transpose Vector
function rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(sum),
    f,
    xs::Union{Adjoint{<:Number,<:AbstractVector},Transpose{<:Number,<:AbstractVector}};
    kwargs...
)
    op = xs isa Adjoint ? adjoint : transpose
    # since summing a vector we don't need to worry about dims which simplifies adjointing
    vector = parent(xs)
    y, vector_sum_pb = rrule(config, sum, f, vector; kwargs...)
    function covector_sum_pb(ȳ)
        s̄um, f̄, v̄ = vector_sum_pb(ȳ)
        return s̄um, f̄, op(v̄)
    end

    return y, covector_sum_pb
end


function rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(sum), f, xs::AbstractArray; dims=:
)
    fx_and_pullbacks = map(x->rrule_via_ad(config, f, x), xs)
    y = sum(first, fx_and_pullbacks; dims=dims)

    pullbacks = last.(fx_and_pullbacks)

    function sum_pullback(ȳ)
        call(f, x) = f(x)  # we need to broadcast this to handle dims kwarg
        f̄_and_x̄s = call.(pullbacks, ȳ)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            sum(first, f̄_and_x̄s)
        end
        x̄s = map(last, f̄_and_x̄s)
        return NoTangent(), f̄, x̄s
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
        return (NoTangent(), NoTangent(), x_thunk)
    end
    return y, sum_abs2_pullback
end

# Fix dispatch for this pidgeon-hole optimization,
# Rules with RuleConfig dispatch with priority over without (regardless of other args).
# and if we don't specify what do do for one that HasReverseMode then it is ambigious
for Config in (RuleConfig, RuleConfig{>:HasReverseMode})
    @eval function rrule(
        ::$Config, ::typeof(sum), ::typeof(abs2), x::AbstractArray{T}; dims=:,
    ) where {T<:Union{Real,Complex}}
        return rrule(sum, abs2, x; dims=dims)
    end
end

#####
##### `prod`
#####

function rrule(::typeof(prod), x::AbstractArray{T}; dims=:) where {T<:CommutativeMulNumber}
    y = prod(x; dims=dims)
    # vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
    function prod_pullback(dy)
        x_thunk = InplaceableThunk(
            # Out-of-place versions
            @thunk if dims === (:)
                ∇prod(x, dy, y)
            elseif any(iszero, x)  # Then, and only then, will ./x lead to NaN
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims(vald, x, dy, y)  # val(Int(dims)) is about 2x faster than Val(Tuple(dims))
            else
                conj.(y ./ x) .* dy
            end
            ,
            # In-place versions -- same branching
            dx -> if dims === (:)
                ∇prod!(dx, x, dy, y)
            elseif any(iszero, x)
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims!(dx, vald, x, dy, y)
            else
                dx .+= conj.(y ./ x) .* dy
            end
            )
        return (NoTangent(), x_thunk)
    end
    return y, prod_pullback
end

function ∇prod_dims(vald::Val{dims}, x, dy, y=prod(x; dims=dims)) where {dims}
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T))
    ∇prod_dims!(dx, vald, x, dy, y)
    return dx
end

function ∇prod_dims!(dx, ::Val{dims}, x, dy, y) where {dims}
    iters = ntuple(d -> d in dims ? tuple(:) : axes(x,d), ndims(x))  # Without Val(dims) this is a serious type instability
    @inbounds for ind in Iterators.product(iters...)
        jay = map(i -> i isa Colon ? 1 : i, ind)
        @views ∇prod!(dx[ind...], x[ind...], dy[jay...], y[jay...])
    end
    return dx
end

function ∇prod(x, dy::Number=1, y::Number=prod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T)) # axes(x) makes MArray on StaticArrays, Array for structured matrices
    ∇prod!(dx, x, dy, y)
    return dx
end

function ∇prod!(dx, x, dy::Number=1, y::Number=prod(x))
    numzero = iszero(y) ? count(iszero, x) : 0
    if numzero == 0  # This can happen while y==0, if there are several small xs
        dx .+= conj.(y ./ x) .* dy
    elseif numzero == 1
        ∇prod_one_zero!(dx, x, dy)
    else
        # numzero > 1, then all first derivatives are zero
    end
    return dx
end

function ∇prod_one_zero!(dx, x, dy::Number=1)  # Assumes exactly one x is zero
    i_zero = 0
    p_rest = one(promote_type(eltype(x), typeof(dy)))
    for i in eachindex(x)
        xi = @inbounds x[i]
        p_rest *= ifelse(iszero(xi), one(xi), conj(xi))
        i_zero = ifelse(iszero(xi), i, i_zero)
    end
    dx[i_zero] += p_rest * dy
    return
end

#####
##### `cumprod`
#####

function rrule(::typeof(cumprod), x::AbstractVector{<:Real}; dims=1)
    y = cumprod(x; dims=dims)  # does nothing unless dims == 1
    function cumprod_pullback_1(dy)
        dx_thunk = InplaceableThunk(
            @thunk if dims == 1
                ∇cumprod(x, dy, y)
            else
                dy
            end
            ,
            dx -> if dims == 1
                ∇cumprod!(dx, x, dy, y)
            else
                dx .+= dy
            end
            )
        return (NO_FIELDS, dx_thunk)
    end
    return y, cumprod_pullback_1
end

function rrule(::typeof(cumprod), x::AbstractArray{<:Real}; dims)
    y = cumprod(x; dims=dims)
    @assert dims isa Integer
    function cumprod_pullback_2(dy)
        dx_thunk = InplaceableThunk(
            @thunk if dims <= ndims(x)
                vald = Val(Int(dims))
                ∇cumprod_dim(vald, x, dy, y)
            else
                dy
            end
            ,
            dx -> if dims <= ndims(x)
                vald = Val(Int(dims))
                ∇cumprod_dim!(dx, vald, x, dy, y)
            else
                dx .+= dy
            end
            )
        return (NO_FIELDS, dx_thunk)
    end
    return y, cumprod_pullback_2
end

function ∇cumprod_dim(vald::Val{dim}, x::AbstractArray, dy=fill!(zero(x),1), y=cumprod(x; dims=dim)) where {dim}
     T = promote_type(eltype(x), eltype(dy))
     dx = fill!(similar(x, T, axes(x)), zero(T))
     ∇cumprod_dim!(dx, vald, x, dy, y)
     return dx
 end

function ∇cumprod_dim!(dx::AbstractArray, ::Val{dim}, x::AbstractArray, dy, y) where {dim}
    iters = ntuple(k -> k==dim ? Ref(:) : axes(x,k), ndims(x))
    for ind in Iterators.product(iters...)
        @views ∇cumprod!(dx[ind...], x[ind...], dy[ind...], y[ind...])
    end
    return dx
end

function ∇cumprod(x::AbstractVector, dy=one(x), y=cumprod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T))  # axes(x) makes MArray on StaticArrays, Array for structured matrices
    ∇cumprod!(dx, x, dy, y)
    return dx
end

function ∇cumprod!(dx::AbstractVector, x::AbstractVector, dy, y)
    lo, hi = firstindex(x), lastindex(x)
    z = something(findfirst(iszero, x), hi+1)
    @inbounds for i in lo:z-1
        ixi = 1/x[i]
        for k in i:z-1
            dx[i] += y[k] * dy[k] * ixi
        end
    end
    @inbounds if z != hi+1
        yk = z==1 ? one(eltype(y)) : y[z-1]  # will be prod(x[j] for j=1:k if j!=z)
        dx[z] += yk * dy[z]
        for k in (z+1):hi
            yk *= x[k]
            dx[z] += yk * dy[k]
        end
    end
    return dx
end
