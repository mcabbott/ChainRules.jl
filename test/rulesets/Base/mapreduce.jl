@testset "Maps and Reductions" begin
    @testset "sum" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            fkwargs = (dims=dims,)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                s = sizes[1:N]
                x = randn(T, s...)
                ẋ = randn(T, s...)
                x̄ = randn(T, s...)
                y = sum(x; dims=dims)
                Δy = randn(eltype(y), size(y)...)
                frule_test(sum, (x, ẋ); fkwargs=fkwargs)
                rrule_test(sum, Δy, (x, x̄); fkwargs=fkwargs)
            end
        end
    end  # sum

    @testset "sum abs2" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            fkwargs = (dims=dims,)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                s = sizes[1:N]
                x, ẋ, x̄ = randn(T, s...), randn(T, s...), randn(T, s...)
                y = sum(abs2, x; dims=dims)
                Δy = randn(eltype(y), size(y)...)
                @testset "frule" begin
                    # can't use frule_test here because it doesn't yet ignore nothing tangents
                    y_ad, ẏ_ad = frule((Zero(), Zero(), ẋ), sum, abs2, x; dims=dims)
                    @test y_ad == y
                    ẏ_fd = jvp(_fdm, z -> sum(abs2, z; dims=dims), (x, ẋ))
                    @test ẏ_ad ≈ ẏ_fd
                end
                @testset "rrule" begin
                    rrule_test(sum, Δy, (abs2, nothing), (x, x̄); fkwargs=fkwargs)
                end
            end
        end
    end  # sum abs2

    # @testset "cumsum" begin
    #     @testset "Array{$T}" for T in [Float64,]
    #         v, vdot, vbar = rand(T, 5), rand(T, 5), rand(T, 5)
    #         m, mdot, mbar = rand(T, 3, 4), rand(T, 3, 4), rand(T, 3, 4)

    #         frule_test(cumsum, (v, vdot))
    #         rrule_test(cumsum, cumsum(v), (v, vbar))

    #         frule_test(cumsum, (m, mdot); fkwargs=(dims=1,))
    #         rrule_test(cumsum, cumsum(m; dims=1), (m, mbar); fkwargs=(dims=1,))

    #         frule_test(cumsum, (m, mdot); fkwargs=(dims=2,))
    #         rrule_test(cumsum, cumsum(m; dims=2), (m, mbar); fkwargs=(dims=2,))
    #     end
    # end

    @testset "prod" begin
        @testset "Array{$T}" for T in [Float64, ComplexF64]
            @testset "size = $sz, dims = $dims" for (sz, dims) in [
                ((12,), :), ((12,), 1),
                ((3,4), 1), ((3,4), 2), ((3,4), :), ((3,4), [1,2]),
                ((3,4,1), 1), ((3,2,2), 3), ((3,2,2), 2:3),
                ]
                x, xdot, xbar = randn(T, sz), randn(T, sz), randn(T, sz)
                # frule_test(prod, (x, xdot); fkwargs=(dims=dims,))
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[1] = 0
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[5] = 0
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[3] = x[7] = 0  # two zeros along some slice, for any dims
                # frule_test(prod, (x, xdot); fkwargs=(dims=dims,))
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                if ndims(x) == 3
                    xp = PermutedDimsArray(x, (3,2,1))  # not a StridedArray
                    xpdot, xpbar = permutedims(xdot, (3,2,1)), permutedims(xbar, (3,2,1))
                    # frule_test(prod, (xp, xpdot); fkwargs=dims)
                    rrule_test(prod, prod(xp; dims=dims), (xp, xpbar); fkwargs=(dims=dims,))
                end
            end
        end
        @testset "Array{Float32}" begin

            [1f-20, 1f-20, 1f-20]
        end
    end # prod
end
