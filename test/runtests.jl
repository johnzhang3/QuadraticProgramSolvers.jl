using QuadraticProgramSolvers
using Test
using LinearAlgebra
##########################################################################################
## check KKT conditions of the randomly generated QPs, make sure the solution is correct
##########################################################################################
@testset "random qp kkt conditions" begin
    for nx in 10:10:100
        for i in 1:10
            n_eq, n_ineq = nx ÷ 2, nx ÷ 2
            qp = rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=i)
            # stationarity
            @test norm(qp.Q * qp.x_sol + qp.q + qp.A' * qp.μ_sol + qp.G' * qp.λ_sol, Inf) < 1e-6
            # primal feasibility
            @test norm(qp.A * qp.x_sol - qp.b, Inf) < 1e-6
            @test norm(qp.G * qp.x_sol - qp.h + qp.s_sol, Inf) < 1e-6
            # dual feasibility
            @test all(qp.λ_sol .>= 0)
            @test all(qp.s_sol .>= 0)
            # complementary slackness
            @test norm(qp.λ_sol .* qp.s_sol, Inf) < 1e-6
        end
    end
end

##########################################################################################
## check the solution of the randomly generated QPs using the primal-dual solver
##########################################################################################
function check_solution(solver::PDIPM, qp::QP, tol=1e-6)
    @test norm(solver.x - qp.x_sol, Inf) < tol
    @test norm(solver.μ - qp.μ_sol, Inf) < tol
    @test norm(solver.λ - qp.λ_sol, Inf) < tol
    @test norm(solver.s - qp.s_sol, Inf) < tol
end

@testset "primal-dual IP on equally mixed random QPs" begin
    for nx in 10:10:100
        for i in 1:10
            n_eq, n_ineq = nx ÷ 2, nx ÷ 2
            qp = rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=i)
            tol = 1e-10
            solver = PDIPM(qp; ip_tol=tol)
            solve!(solver)
            check_solution(solver, qp, 1e-6)
        end
    end
end

##########################################################################################
## check the solution of the randomly generated QPs using the log-domain solver
##########################################################################################
function check_solution(solver::LDIPM, qp::QP, tol=1e-6)
    @test norm(solver.x - qp.x_sol, Inf) < tol
    @test norm(solver.μ - qp.μ_sol, Inf) < tol
    @test norm(sqrt(solver.κ) * exp.(solver.σ) - qp.λ_sol, Inf) < tol
    @test norm(sqrt(solver.κ) * exp.(-solver.σ) - qp.s_sol, Inf) < tol
end
@testset "log domain IP on equally mixed random QPs" begin
    for nx in 10:10:100
        for i in 1:10
            n_eq, n_ineq = nx ÷ 2, nx ÷ 2
            qp = rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=i)
            tol = 1e-10
            solver = LDIPM(qp; ip_tol=tol)
            solve!(solver)
            check_solution(solver, qp, 1e-6)
        end
    end
end