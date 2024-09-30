"""
    convex Quadratic Program of the form:
    min 1/2 x'Qx + q'x      min 1/2 x'Qx + q'x
    s.t. Ax - b = 0     ==> s.t. Ax - b = 0
         Gx - h <= 0             Gx - h + s = 0
                                 s >= 0

    the lagrangian is:
    L(x, μ, λ) = 1/2 x'Qx + q'x + μ'(Ax - b) + λ'(Gx - h)
    where μ and λ are the lagrange multipliers for the equality and inequality constraints

    the KKT conditions are:
    1. ∇L/∇x = Qx + q + A'μ + G'λ = 0
    2. Ax - b = 0
    3. Gx - h <= 0
    4. λ >= 0
    5. λ'(Gx + h) = 0

"""
mutable struct QP
    Q::AbstractMatrix
    q::Vector
    A::AbstractMatrix
    b::Vector
    G::AbstractMatrix
    h::Vector

    x_sol::Union{Vector, Nothing} # primal solution
    μ_sol::Union{Vector, Nothing} # lagrange multiplier for equality constraints
    λ_sol::Union{Vector, Nothing} # lagrange multiplier for inequality constraints
    s_sol::Union{Vector, Nothing} # slack variable    

    nx::Int # number of decision variables
    n_eq::Int # number of equality constraints
    n_ineq::Int # number of inequality constraints
    
    function QP(; Q::Matrix, q::Vector, A::Matrix, b::Vector, G::Matrix, h::Vector, 
        x_sol::Union{Vector, Nothing}=nothing, μ_sol::Union{Vector, Nothing}=nothing, 
        λ_sol::Union{Vector, Nothing}=nothing, s_sol::Union{Vector, Nothing}=nothing)
        return new(Q, q, A, b, G, h, x_sol, μ_sol, λ_sol, s_sol,
         size(Q, 1), size(A, 1), size(G, 1))
    end
end

function rand_qp(;nx=10, n_eq=5, n_ineq=5, seed=1)
    Random.seed!(seed)
    
    Q = randn(nx, nx)
    Q = Q' * Q
    
    A = randn(n_eq, nx)
    G = randn(n_ineq, nx)
    
    μ = randn(n_eq)
    s = abs.(randn(n_ineq))
    λ = abs.(randn(n_ineq))
    for i in 1:n_ineq
        if rand() > 0.5
            s[i] = 0
        else
            λ[i] = 0
        end 
        
    end

    x = randn(nx) # solution
    b = A * x
    h = G * x + s
    
    q = -A' * μ - G' * λ - Q * x

    return QP(Q=Q, q=q, A=A, b=b, G=G, h=h, x_sol=x, μ_sol=μ, λ_sol=λ, s_sol=s)
end