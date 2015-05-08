from numpy import *
from numpy.linalg import qr


def GR(a):
    """
    Gram-Schmidt Verfahren

    Keyword Arguments:
    a -- (m x n) Matrix

    Returns: q, r
    q -- Matrix Q
    r -- Matrix R
    """

    #
    # implementiere das Gram-Schmidt Verfahren hier  #
    #
    q = None  # Platzhalter
    r = None
    return q, r


def GRmod(a):
    """
    modifiziertes Gram-Schmidt Verfahren
    Keyword Arguments:
    a -- (m x n) Matrix

    Returns: q, r
    q -- Matrix Q
    r -- Matrix R
    """

    #
    # implementiere das modifizierte Gram-Schmidt Verfahren hier #
    #
    q = None  # Platzhalter
    r = None
    return q, r


# main

# Matrix Definition
n = 50
m = 50
Z = zeros((m, n))
for i in xrange(m):
    for j in xrange(n):
        Z[i, j] = 1 + min(i, j)

# numpy QR-implementation (als Vergleich)
q0, r0 = qr(Z, mode='full')

## Berechne hier die Guete deines oben implementierten Gram-Schmidt Verfahren
#q2, r2 = GR(Z)

## Guete des modifizierten Gram-Schmidt Verfahren
#q3, r3 = GRmod(Z)

## print statement
print("numpys qr liefert:         %.10e" % (dot(q0.T, q0) - eye(n)).max())
# print("Gram-Schmidt liefert:      %.10e" % (dot(q2.T, q2) - eye(n)).max())
# print("mod. Gram-Schmidt liefert: %.10e" % (dot(q3.T, q3) - eye(n)).max())
