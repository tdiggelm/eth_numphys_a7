from numpy import *
from numpy.linalg import solve, norm
from matplotlib.pyplot import *

###################
# Unteraufgabe a) #
###################

def row_2_step(f, Jf, yi, h):
    r"""Rosenbrock-Wanner Methode der Ordnung 2

    Input:
    f:   Die rhs Funktion f(x)
    Jf:  Jacobi Matrix J(x) der Funktion: R^(nx1) -> R^(nxn)
    yi:  Aktueller Wert y_i zur Zeit ti
    h:   Schrittweite

    Output:
    yip1:  Zeitpropagierter Wert y(t+h): R^(nx1)
    """
    yi = atleast_2d(yi)
    n = yi.shape[0]
    yip1 = zeros_like(yi)

    ####################################################
    #                                                  #
    # TODO: Implementieren Sie die ROW-2 Methode hier. #
    #                                                  #
    ####################################################
    
    a = 1/(2+sqrt(2))
    J = Jf(yi)
    I = identity(n)
    
    k1 = solve((I - a*h*J), f(yi))
    k2 = solve((I - a*h*J), f(yi+h*0.5*k1) - a*h*J.dot(k1))
    yip1 = yi + h*k2

    return yip1


def row_3_step(f, Jf, yi, h):
    r"""Rosenbrock-Wanner Methode der Ordnung 3

    Input:
    f:   Die rhs Funktion f(x): R^(nx1) -> R^(nx1)
    Jf:  Jacobi Matrix J(x) der Funktion: R^(nx1) -> R^(nxn)
    yi:  Aktueller Wert y_i zur Zeit ti
    h:   Schrittweite

    Output:
    yip1:  Zeitpropagierter Wert y(t+h): R^(nx1)
    """
    yi = atleast_2d(yi)
    n = yi.shape[0]
    yip1 = zeros_like(yi)

    ####################################################
    #                                                  #
    # TODO: Implementieren Sie die ROW-3 Methode hier. #
    #                                                  #
    ####################################################

    a = 1/(2+sqrt(2))
    d31 = -(4+sqrt(2))/(2+sqrt(2))
    d32 = (6+sqrt(2))/(2+sqrt(2))
    J = Jf(yi)
    I = identity(n)
    
    k1 = solve((I - a*h*J), f(yi))
    k2 = solve((I - a*h*J), f(yi+h*0.5*k1) - a*h*J.dot(k1))
    k3 = solve((I - a*h*J), f(yi+h*k2) - d31*h*J.dot(k1) - d32*h*J.dot(k2))
    yip1 = yi + h/6*(k1+4*k2+k3)

    return yip1


def constructor(stepalg):
    r"""
    Input:
    stepalg:  Methode um einen Zeitschritt zu machen

    Output:
    stepper:  Einen Integrator der die gegebene Methode anwendet.
    """

    def stepper(f, Jf, t0, y0, h, N):
        r"""
        Input:
        f:   Die rhs Funktion f(x): R^(nx1) -> R^(nx1)
        Jf:  Jacobi Matrix J(x) der Funktion: R^(nx1) -> R^(nxn)
        t0:  Startzeitpunt: R^1
        y0:  Anfangswert y(t0): R^(nx1)
        h:   Schrittweite
        N:   Anzahl Schritte

        Output:
        t:    Zeitpute ti: R^N
        sol:  Loesung y(ti): R^(nxN)
        """
        # Copy and reshape input
        ti = atleast_1d(t0).copy().reshape(1)
        yi = atleast_1d(y0).copy().reshape(-1,1)
        # Wrap function calls for shape consistency
        n = yi.shape[0]
        ff = lambda y: f(y).reshape(n,1)
        Jff = lambda y: Jf(y).reshape(n,n)
        # Collect results
        t = [ti]
        sol = [yi]
        # Time iteration
        for i in xrange(1,N+1):
            ti = ti + h
            yi = stepalg(ff, Jff, yi, h)
            t.append(ti)
            sol.append(yi)
        # Stack together results
        return hstack(t).reshape(-1), hstack(sol).reshape(n,-1)

    return stepper


# Construct integrators for ROW-2 and ROW-3
row_2 = constructor(row_2_step)
row_3 = constructor(row_3_step)


###################
# Unteraufgabe b) #
###################

def aufgabe_b():
    print(" Aufgabe b)")
    # Logistic ODE
    c = 0.01
    l = 80
    f = lambda y: l*y*(1 - y)
    Jf = lambda y: l - 2*l*y

    sol = lambda t: (c*exp(l*t)) / (1 - c + c*exp(l*t))
    t0 = 0.0
    y0 = c

    T = 2.0
    N = 100
    h = T/float(N)

    ##################################################
    #                                                #
    # TODO: Loesen Sie die logistische Gleichung mit #
    #       den ROW Methoden und plotten Sie die     #
    #       Loesung und den Fehler.                  #
    #                                                #
    ##################################################

    
    def find_lambda(l0=55, le=60, dl=0.1, max_err=0.05):
        for l in arange(l0,le,dl):
            f = lambda y: l*y*(1 - y)
            Jf = lambda y: l - 2*l*y
            sol = lambda t: (c*exp(l*t)) / (1 - c + c*exp(l*t))
            t, y = constructor(row_2_step)(f, Jf, t0, y0, h, N)
            err = max(abs(y[0] - sol(t)))
            if err > max_err:
                return True, l, err
        return False, le, 0
        
    found_lambda, lf, err = find_lambda()
    if found_lambda:
        print("Error for row2 is becoming larger than 0.05 for lambda=%s: err=%s." % (lf, err))
    
    # TODO: row3 plot implement, plot row2 for multiple values of lambda, fix title
    
    ax1 = figure().add_subplot(111)
    t_row2, y_row2 = constructor(row_2_step)(f, Jf, t0, y0, h, N)
    y_sol = sol(t_row2)
    ax1.plot(t_row2, y_row2[0], alpha=0.7, color="blue", label=r"$y_{row2}(t)$")
    ax1.plot(t_row2, y_sol, "--", linewidth=1.5, alpha=0.7, color="gray", label=r"$y_{sol}(t)$")
    legend(loc="lower right")
    ax2 = ax1.twinx()
    y_err = abs(y_row2[0] - y_sol)
    ax2.plot(t_row2, y_err, alpha=0.8, color="red", label=r"$err:=|y_{row2}(t)-y_{sol}(t)|$")
    legend(loc="upper right")
    grid(True)
    title(r"Integrator ROW2 $\lambda:=%s$" % l)
    xlabel(r"$t$")
    savefig("plot_row2_l%s.png" % l)


    ax1 = figure().add_subplot(111)    
    t_row3, y_row3 = constructor(row_3_step)(f, Jf, t0, y0, h, N)
    y_sol = sol(t_row3)
    ax1.plot(t_row3, y_row3[0], alpha=0.7, color="blue", label=r"$y_{row3}(t)$")
    ax1.plot(t_row3, y_sol, "--", alpha=0.7, linewidth=1.5, color="gray", label=r"$y_{sol}(t)$")
    legend(loc="lower right")
    ax2 = ax1.twinx()
    y_err = abs(y_row3[0] - y_sol)
    ax2.plot(t_row3, y_err, alpha=0.8, color="red", label=r"$err:=|y_{row3}(t)-y_{sol}(t)|$")
    legend(loc="upper right")
    grid(True)
    title(r"Integrator ROW2 $\lambda:=%s$" % l)
    xlabel(r"$t$")
    savefig("plot_row3_l%s.png" % l)

    
#    t, y = constructor(row_3_step)(f, Jf, t0, y0, h, N)
#    plot(t, y[0], label=r"$y_{row3}(t)$")

    
###################
# Unteraufgabe c) #
###################

def aufgabe_c():
    print(" Aufgabe c)")
    # Logistic ODE
    c = 0.01
    l = 10.0
    f = lambda y: l*y*(1-y)
    Jf = lambda y: l- 2*l*y

    sol = lambda t: (c*exp(l*t)) / (1 - c + c*exp(l*t))
    t0 = 0.0
    y0 = c
    T = 2.0

    # Different number steps
    steps = 2**arange(4,13)

    # Storage for solution values
    datae = []
    data2 = []
    data3 = []

    for N in steps:

        ##################################################
        #                                                #
        # TODO: Loesen Sie die logistische Gleichung mit #
        #       den ROW Methoden.                        #
        #                                                #
        ##################################################
        pass

    datae = array(datae)
    data2 = array(data2)
    data3 = array(data3)

    err2 = ones_like(steps, dtype=float)
    err3 = ones_like(steps, dtype=float)

    ##################################################
    #                                                #
    # TODO: Berechnen Sie den Fehler gegenueber der  #
    #       exakten Loesung zum Endzeitpunkt T.      #
    #                                                #
    ##################################################

    figure()
    loglog(steps, err2, "b-o", label="ROW-2")
    loglog(steps, err3, "g-o", label="ROW-3")
    loglog(steps, 1e-5*(float(T)/steps)**2, "-k", label="$O(N^-2)$")
    loglog(steps, 1e-5*(float(T)/steps)**3, "--k", label="$O(N^-3)$")
    grid(True)
    xlim(steps.min(), steps.max())
    legend(loc="lower left")
    xlabel(r"Number of steps $N$")
    ylabel(r"Absolute Error at $T = %.1f$" % T)
    savefig("convergence_row.png")


###################
# Unteraufgabe d) #
###################

def odeintadapt(Psilow, Psihigh, T, y0, fy0=None, h0=None, hmin=None, reltol=1e-2, abstol=1e-4):
    r"""Adaptive Integrator for stiff ODEs and Systems
    based on two suitable methods of different order.

    Input:
    Psilow:   Integrator of low order
    Psihigh:  Integrator of high order
    T:        Endtime: R^1
    y0:       Initial value: R^(nx1)
    fy0:      The value f(y0): R^(nx1) used to estimate initial timestep size
    h0:       Initial timestep (optional)
    hmin:     Minimal timestep (optional)
    reltol:   Relative tolerance (optional)
    abstol:   Absolute Tolerance (optional)

    Output:
    t:    Timesteps: R^K with K the number of steps performed
    y:    Solution at the timesteps: R^(nxK)
    rej:  Time of rejected timesteps: R^G with G the number of rejections
    ee:   Estimated error: R^K
    """
    # Heuristic choice of initial timestep size
    if h0 is None:
        h0 = T / (100.0*(norm(fy0) + 0.1))
    if hmin is None:
        hmin = h0 / 10000.0

    # Initial values
    yi = atleast_2d(y0).copy()
    ti = 0.0
    hi = h0
    n = yi.shape[0]

    # Storage
    t = [ti]
    y = [yi]
    rej = []
    ee = [0.0]

    while ti < T and hi > hmin:

        #########################################################
        #                                                       #
        # TODO: Implementieren Sie hier die adaptive Strategie. #
        #                                                       #
        #########################################################
        pass

    return array(t).reshape(-1), array(y).T.reshape(n,-1), array(rej), array(ee)


###################
# Unteraufgabe e) #
###################

def aufgabe_e():
    print(" Aufgabe e)")
    # Logistic ODE
    c = 0.01
    l = 50.0
    f = lambda y: l*y*(1-y)
    Jf = lambda y: l- 2*l*y

    sol = lambda t: (c*exp(l*t)) / (1 - c + c*exp(l*t))
    y0 = sol(0.0)
    T = 2.0

    nsteps = 0

    ########################################################################
    #                                                                      #
    # TODO: Testen Sie die adaptive Methode an der logistischen Gleichung. #
    #       Berechnen Sie die Anzahl der Zeitschritte. Plotten Sie die     #
    #       Loesung und den Fehler.                                        #
    #                                                                      #
    ########################################################################

    print("Es werden %d Zeitschritte benoetigt" % nsteps)


###################
# Unteraufgabe f) #
###################

def aufgabe_f():
    print(" Aufgabe f)")
    # Test case
    tau = 4*pi/3.0
    R = array([[sin(tau), cos(tau)],[-cos(tau), sin(tau)]])
    D = array([[-101.0, 0.0],[0.0, -1.0]])
    A = dot(R.T, dot(D, R))

    f = lambda y: dot(A, y)
    Jf = lambda y: A
    y0 = array([[1.0],
                [1.0]])
    T = 1.0

    ###################################################################
    #                                                                 #
    # TODO: Loesen Sie das Gleichungssystem mit der adaptiven Methode #
    #       und Plotten Sie die Loesung.                              #
    #                                                                 #
    ###################################################################


###################
# Unteraufgabe g) #
###################

def aufgabe_g():
    print(" Aufgabe g)")
    # Logistic ODE with y squared
    l = 500.0
    f = lambda y: l*y**2*(1-y**2)
    Jf = lambda y: l*(2*y-4*y**3)
    y0 = 0.01
    T = 0.5

    hmin = 1.0
    nrsteps = 0.0

    ##############################################################
    #                                                            #
    # TODO: Loesen Sie die steife Gleichung und plotten Sie      #
    #       die Loesung sowie die Groesse der Zeitschritte gegen #
    #       die Zeit.                                            #
    #                                                            #
    #       Wie viele Zeitschritte benoetigt dieses Verfahren?   #
    #       Was ist der kleinste Zeitschritt?                    #
    #       Wie viele Zeitschritte dieser Groesse wuerde ein     #
    #       nicht-adapives Verfahren benoetigen?                 #
    #                                                            #
    ##############################################################


    print("Minimal steps size: %f" % hmin)
    print("Number of adaptive steps: %i" % nrsteps)
    print("Number of non-adaptive steps: %.2f" % (T/hmin))




if __name__ == "__main__":
    # Run all subtasks
    aufgabe_b()
    aufgabe_c()
    aufgabe_e()
    aufgabe_f()
    aufgabe_g()
