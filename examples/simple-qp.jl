using QuadraticProgramSolvers

qp = rand_qp(nx=2, n_eq=2, n_ineq=2, seed=1);

solver = PDIPM(qp, verbose=false)
solve!(solver)

solver = LDIPM(qp, verbose=true)
solve!(solver)

solver = ADMM(qp, verbose=true)
solve!(solver)
