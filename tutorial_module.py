from petsc4py import PETSc
import numpy as np

# ----------------------------
# Manufactured solution (2D)
# u = 2t + x^2 + y^2
# ----------------------------
def u_exact(x, y, t):
    return 2.0 * t + x*x + y*y


def forcing(x, y, t):
    # u_t = 2
    # Δu = 4
    # f = 2 - 4 = -2
    return -2.0


# ----------------------------
# Build 2D Laplacian (FD grid)
# ----------------------------
def build_laplacian(N, h):
    A = PETSc.Mat().createAIJ([N*N, N*N])
    A.setUp()

    def idx(i, j):
        return i*N + j

    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            A[row, row] = 4.0 / (h*h)

            if i > 0:
                A[row, idx(i-1, j)] = -1.0 / (h*h)
            if i < N-1:
                A[row, idx(i+1, j)] = -1.0 / (h*h)
            if j > 0:
                A[row, idx(i, j-1)] = -1.0 / (h*h)
            if j < N-1:
                A[row, idx(i, j+1)] = -1.0 / (h*h)

    A.assemble()
    return A


# ----------------------------
# Main solver
# ----------------------------
def main():
    comm = PETSc.COMM_WORLD

    N = 20
    T = 0.1
    dt = 0.01
    steps = int(T / dt)

    h = 1.0 / (N - 1)

    # Laplacian operator
    A = build_laplacian(N, h)

    # Identity matrix
    I = PETSc.Mat().createAIJ([N*N, N*N])
    I.setUp()
    I.assemble()

    # System matrix: I + dt*A
    K = I.copy()
    K.axpy(dt, A)

    # Solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(K)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setFromOptions()

    # Vectors
    u = PETSc.Vec().createSeq(N*N)
    u_new = u.duplicate()
    rhs = u.duplicate()

    u.set(0.0)

    # time loop
    t = 0.0

    for _ in range(steps):
        t += dt

        u_arr = u.getArray()  # cache for speed

        for i in range(N):
            for j in range(N):
                row = i*N + j
                x = i * h
                y = j * h

                rhs[row] = u_arr[row] + dt * forcing(x, y, t)

        rhs.assemble()

        ksp.solve(rhs, u_new)

        u = u_new.copy()

    # ----------------------------
    # error computation
    # ----------------------------
    u_arr = u.getArray()

    err = 0.0
    for i in range(N):
        for j in range(N):
            x = i * h
            y = j * h
            row = i*N + j

            err += (u_arr[row] - u_exact(x, y, t))**2

    err = np.sqrt(err / (N*N))

    if comm.rank == 0:
        print("Final time:", t)
        print("L2 error:", err)


if __name__ == "__main__":
    main()