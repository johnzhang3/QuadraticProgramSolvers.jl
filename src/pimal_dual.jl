"""
    Primual-dual interior point method (PDIPM) solver
"""
mutable struct PDIPM <: AbstractSolver
    
    qp::QP # QP instance

    x::Vector # primal variable
    μ::Vector # lagrange multiplier for equality constraints
    λ::Vector # lagrange multiplier for inequality constraints
    s::Vector # slack variable
    κ::Float64 # central path parameter
    
    # settings
    ip_tol::Float64 # tolerance for the KKT residual
    max_ip_iter::Int # maximum number of iterations
    max_ls_iter::Int # maximum number of iterations for the line search
    verbose::Bool # print the iteration information

end

function PDIPM(qp::QP; x::Vector=zeros(qp.nx), μ::Vector=zeros(qp.n_eq), λ::Vector=ones(qp.n_ineq), s::Vector=ones(qp.n_ineq),
    κ::Float64=1.0, ip_tol::Float64=1e-8, max_ip_iter::Int=20, 
    max_ls_iter::Int=10,  verbose::Bool=false)
    
    return PDIPM(qp, x, μ, λ, s, κ, ip_tol, max_ip_iter, 
        max_ls_iter, verbose)
end

#TODO: initialization?

function kkt_residual(solver::PDIPM)
    return [solver.qp.Q * solver.x + solver.qp.q + solver.qp.A' * solver.μ + solver.qp.G' * solver.λ;
            solver.qp.A * solver.x - solver.qp.b;
            solver.qp.G * solver.x + solver.s - solver.qp.h;
            solver.λ .* solver.s]
end

function kkt_residual_relaxed(solver::PDIPM)
    return [solver.qp.Q * solver.x + solver.qp.q + solver.qp.A' * solver.μ + solver.qp.G' * solver.λ;
            solver.qp.A * solver.x - solver.qp.b;
            solver.qp.G * solver.x + solver.s - solver.qp.h;
            solver.λ .* solver.s .- solver.κ]
end

function kkt_residual_jacobian(solver::PDIPM)
    Q = solver.qp.Q
    A = solver.qp.A
    G = solver.qp.G
    s = solver.s
    λ = solver.λ
    S = diagm(s)
    Λ = diagm(λ)

    return [
        Q  A'  G'  zeros(size(G, 2), size(S, 1));
        A  zeros(size(A, 1), size(A, 1))  zeros(size(A, 1), size(G, 1))  zeros(size(A, 1), size(S, 1));
        G  zeros(size(G, 1), size(A, 1))  zeros(size(G, 1), size(G, 1))  I(size(G, 1));
        zeros(size(S, 1), size(Q, 2))  zeros(size(S, 1), size(A, 1))  S  Λ
    ]
end

function linesearch(solver::PDIPM, Δλ::Vector, Δs::Vector)
    
    α = 1.0
    for i in 1:solver.max_ls_iter
        λ = solver.λ + α * Δλ
        s = solver.s + α * Δs
        if all(λ .>= 0) && all(s .>= 0)
            return α
        end
        α /= 2
    end
    @warn "linesearch failed"
    return α
    
end

function solve!(solver::PDIPM)

    z = [solver.x; solver.μ; solver.λ; solver.s]

    for i in 1:solver.max_ip_iter
        # newton_solve!(solver)
        res = kkt_residual(solver)
        if norm(res, Inf) < solver.ip_tol
            return nothing
        end 
        ∇ = kkt_residual_jacobian(solver)
        Δ_aff = ∇ \ - res

        Δλ_aff = Δ_aff[solver.qp.nx+solver.qp.n_eq+1:solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq]
        Δs_aff = Δ_aff[solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq+1:end]
        
        α = linesearch(solver, Δλ_aff, Δs_aff)

        # compute the centring-plus-corrector direction
        solver.κ = dot(solver.s, solver.λ) / solver.qp.n_ineq
        μ = (((solver.s + α*Δs_aff)'*(solver.λ + α*Δλ_aff))/dot(solver.s, solver.λ))^3

        rhs = [zeros(solver.qp.nx+solver.qp.n_eq + solver.qp.n_ineq); solver.κ*μ .- Δs_aff .* Δλ_aff]
        Δ_cc = ∇ \ rhs

        Δ = Δ_aff + Δ_cc
        Δλ = Δ[solver.qp.nx+solver.qp.n_eq+1:solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq]
        Δs = Δ[solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq+1:end]
        α = linesearch(solver, Δλ, Δs)
        α = min(1.0, 0.99*α)

        z += α * Δ
        
        solver.x = z[1:solver.qp.nx];
        solver.μ = z[solver.qp.nx+1:solver.qp.nx+solver.qp.n_eq];
        solver.λ = z[solver.qp.nx+solver.qp.n_eq+1:solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq];
        solver.s = z[solver.qp.nx+solver.qp.n_eq+solver.qp.n_ineq+1:end]

        if solver.verbose
            println("iter: $i, norm(res): $(norm(res, Inf)), κ: $(solver.κ)")
        end
        
    end
    @warn "PDIPM did not converge"
    return nothing
end
