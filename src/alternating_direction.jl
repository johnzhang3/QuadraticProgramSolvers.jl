"""
    Alternating Direction Method of Multipliers (ADMM) Solver
"""

mutable struct ADMM <: AbstractSolver
    Q::AbstractMatrix
    q::Vector
    A::AbstractMatrix
    l::Vector
    u::Vector

    x::Vector # primal variable
    x̃::Vector # spliting variable for x
    μ::Vector # lagrange multiplier for equality constraints
    λ::Vector # lagrange multiplier for inequality constraints
    z::Vector # spliting variable for constraints
    ρ::Float64 # penalty parameter
    ρ_min::Float64 # minimum penalty parameter
    ρ_max::Float64 # maximum penalty parameter
    σ::Float64 # penalty parameter for x̃
    tol::Float64 # solver tolerance
    max_iter::Int # maximum number of iterations
    verbose::Bool # print the iteration information
end

function ADMM(qp::QP; x::Vector=zeros(qp.nx), μ::Vector=zeros(qp.nx), λ::Vector=zeros(qp.n_eq+qp.n_ineq), z::Vector=zeros(qp.n_eq+qp.n_ineq),
    ρ::Float64=0.1, ρ_min::Float64=1e-6, ρ_max::Float64=1e6,
    σ::Float64=1e-6, tol::Float64=1e-6,
    max_iter::Int=100, verbose::Bool=false)
    # convert QP to OSQP format
    Q = qp.Q
    q = qp.q
    A = [qp.A; qp.G]
    u = [qp.b; qp.h]
    l = [qp.b; -Inf * ones(qp.n_ineq)]
    x̃ = zeros(qp.nx)
    return ADMM(Q, q, A, l, u,
        x, x̃, μ, λ, z, ρ, ρ_min, ρ_max,
        σ, tol,
         max_iter, verbose) 
end

primal_residual(solver::ADMM) = solver.A * solver.x - solver.z

dual_residual(solver::ADMM) = solver.Q * solver.x + solver.q + solver.A' * solver.λ

function solve!(solver::ADMM)
    if solver.verbose
        @printf("ADMM (OSQP) solver\n")
        @printf("iter         J            ρ            |r_p|∞          |r_d|∞     \n")
        @printf("------------------------------------------------------------------------------------\n")
    end
    Q, q, A, l, u = solver.Q, solver.q, solver.A, solver.l, solver.u
    σ = solver.σ
    ρ = solver.ρ
    for i in 1:solver.max_iter
        # if solver.verbose && i % 25 == 0
        #     @printf("%3d  %12.6e  %12.6e  %12.6e  %12.6e\n", i, cost(solver), solver.ρ, norm(primal_residual(solver), Inf), norm(dual_residual(solver), Inf))
        # end

        # if norm(primal_residual(solver), Inf) < solver.tol && norm(dual_residual(solver), Inf) < solver.tol
        #     return nothing
        # end
        if i % 25 == 0
            if solver.verbose
                @printf("%3d  %12.6e  %12.6e  %12.6e  %12.6e\n", i, cost(solver), solver.ρ, norm(primal_residual(solver), Inf), norm(dual_residual(solver), Inf))
            end

            ρ = ρ * sqrt(norm(primal_residual(solver), Inf) / norm(dual_residual(solver), Inf))

            solver.ρ = max(solver.ρ_min, min(solver.ρ_max, ρ))

            if norm(primal_residual(solver), Inf) < solver.tol && norm(dual_residual(solver), Inf) < solver.tol
                return nothing
            end
            
        end
        x, x̃, μ, λ, z, ρ = solver.x, solver.x̃, solver.μ, solver.λ, solver.z, solver.ρ

        # update x̃
        x̃ = (Q + ρ * A' * A + σ*I) \ (ρ * A' * z + σ*x - μ - A'λ - q)
        # update x
        x = x̃ + μ/σ
        # update z
        z = max.(l, min.(u, A*x̃ + λ/ρ))
        # update λ
        λ = λ + ρ * (A*x - z)
        # update μ
        μ = μ + σ * (x̃ - x)

        solver.x, solver.x̃, solver.μ, solver.λ, solver.z, solver.ρ = x, x̃, μ, λ, z, ρ

    end
    @warn "ADMM did not converge"
end

