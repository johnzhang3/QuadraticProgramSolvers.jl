module QuadraticProgramSolvers
    using Random
    using LinearAlgebra
    using Printf
    abstract type AbstractSolver end
    cost(solver::AbstractSolver) = 0.5 * dot(solver.x, solver.Q * solver.x) + dot(solver.q, solver.x)

    include("utils.jl")
    export QP, rand_qp

    include("pimal_dual.jl")
    export PDIPM, solve!

    include("log_domain.jl")
    export LDIPM

    # TODO Agumented Lagrangian

    include("alternating_direction.jl")
    export ADMM

    
end
