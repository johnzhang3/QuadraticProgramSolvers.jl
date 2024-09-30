mutable struct LDIPM <: AbstractSolver
    qp::QP # stores QP data

    # variables
    x::Vector # primal variables
    μ::Vector # lagrange multiplier for equality constraints
    σ::Vector # lagrange multiplier for inequality constraints in the log domain

    # settings
    κ::Float64   # central path parameter
    ip_tol::Float64 # tolerance for the KKT residual
    max_ip_iter::Int # maximum number of iterations
    α::Float64 # central path parameter reduction
    newton_tol::Float64 # tolerance for the Newton's method for the perturbed KKT system
    max_newton_iter::Int # maximum number of iterations for the Newton's method
    verbose::Bool # print the iteration information

end

function LDIPM(qp::QP; x::Vector=zeros(qp.nx), μ::Vector=zeros(qp.n_eq), σ::Vector=zeros(qp.n_ineq),
    ρ::Float64=1.0, ip_tol::Float64=1e-8, max_ip_iter::Int=20, κ::Float64=0.1, newton_tol::Float64=1e-8, max_newton_iter::Int=10,
    verbose::Bool=false)
    
    # TODO: some setup stuff 
    return LDIPM(qp, x, μ, σ, ρ, ip_tol, max_ip_iter, κ, newton_tol, max_newton_iter, verbose)
end

function kkt_residual(solver::LDIPM)
    s = sqrt(solver.κ) * exp.(-solver.σ)
    λ = sqrt(solver.κ) * exp.(solver.σ)
    return [solver.qp.Q * solver.x + solver.qp.q + solver.qp.A' * solver.μ + solver.qp.G' * λ;
            solver.qp.A * solver.x - solver.qp.b;
            solver.qp.G * solver.x + s - solver.qp.h;
            λ .* s]
end

function kkt_residual_relaxed(solver::LDIPM)
    return [
        solver.qp.Q * solver.x + solver.qp.q + solver.qp.A' * solver.μ + solver.qp.G' * sqrt(solver.κ) * exp.(solver.σ);
        solver.qp.A * solver.x - solver.qp.b;
        solver.qp.G * solver.x + sqrt(solver.κ) * exp.(-solver.σ) - solver.qp.h
        ]
end

function kkt_residual_jacobian(solver::LDIPM)
    return [
        solver.qp.Q solver.qp.A' solver.qp.G' * diagm(sqrt(solver.κ) * exp.(solver.σ));
        solver.qp.A zeros(solver.qp.n_eq, solver.qp.n_eq) zeros(solver.qp.n_eq, solver.qp.n_ineq);
        solver.qp.G zeros(solver.qp.n_ineq, solver.qp.n_eq) -diagm(sqrt(solver.κ) * exp.(-solver.σ))
    ]
end

function newton_solve!(solver::LDIPM)
    z = [solver.x; solver.μ; solver.σ]
    for i in 1:solver.max_newton_iter
        
        res = kkt_residual_relaxed(solver)
        if solver.verbose
            println("iter: $i, norm(res): $(norm(res, Inf))")
        end
        
        if norm(res, Inf) < solver.newton_tol
            return nothing
        end
        ∇ = kkt_residual_jacobian(solver)
        z -= ∇ \ res
        solver.x = z[1:solver.qp.nx]; 
        solver.μ = z[solver.qp.nx+1:solver.qp.nx+solver.qp.n_eq]; 
        solver.σ = z[solver.qp.nx+solver.qp.n_eq+1:end]
    end
    @warn "Newton's method did not converge" 
    return nothing
end

function solve!(solver::LDIPM)
    for i in 1:solver.max_ip_iter
        newton_solve!(solver)
        res = kkt_residual(solver)
        if solver.verbose
            println("iter: $i, norm(res): $(norm(res, Inf)), κ: $(solver.κ)")
        end
        if norm(res, Inf) < solver.ip_tol
            return nothing
        end
        
        solver.κ *= solver.α
    end
    @warn "LDIPM did not converge"
    return nothing
end