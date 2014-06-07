"""
Utitility functions for working with operator transformations in
sympy.physics.quantum.
"""
from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.boson import *
from sympy.physics.quantum.fermion import *
from sympy.physics.quantum.operatorordering import *

debug = False

#
# IPython notebook related functions
#
from IPython.display import display_latex
from IPython.display import Latex

def show_first_few_terms(e, n=10):
    if isinstance(e, Add):
        e_args_trunc = e.args[0:n]
        e = Add(*(e_args_trunc))
    
    return Latex("$" + latex(e).replace("dag", "dagger") + r"+ \dots$")


#
# Functions for use with sympy.physics.quantum
#
def recursive_commutator(a, b, n=1):
    return Commutator(a, b) if n == 1 else Commutator(a, recursive_commutator(a, b, n-1))


def bch_expansion(A, B, N=10):
    """
    Baker–Campbell–Hausdorff formula:
    
    e^{A} B e^{-A} = B + 1/(1!)[A, B] + 1/(2!)[A, [A, B]] + 1/(3!)[A, [A, [A, B]]] + ...
                   = B + Sum_n^N 1/(n!)[A, B]^n
                   
    Truncate the sum at N terms.
    """
    e = B
    for n in range(1, N):
        e += recursive_commutator(A, B, n=n) / factorial(n)
    
    return e


def split_coeff_operator(e):
    """
    Split a product of coefficients, commuting variables and quantum operators
    into two factors containing the commuting factors and the quantum operators,
    resepectively.
    
    Returns:
    c_factor, o_factors:
        Commuting factors and noncommuting (operator) factors
    """
    if isinstance(e, Symbol):
        return e, 1

    if isinstance(e, Operator):
        return 1, e

    if isinstance(e, Mul):
        c_args = []
        o_args = []    

        for arg in e.args:
            if isinstance(arg, Operator):
                o_args.append(arg)
            elif isinstance(arg, Pow):
                c, o = split_coeff_operator(arg.base)

                if c != 1:
                    c_args.append(c ** arg.exp)
                if o != 1:
                    o_args.append(o ** arg.exp)
            else:
                c_args.append(arg)
                
        return Mul(*c_args), Mul(*o_args)

    if debug:
        print("Warning: Unrecongized type of e: %s" % type(e))
    return None, None


def extract_operator_products(e, independent=False):
    """
    Return a list of unique normal-ordered quantum operator products in the
    expression e.
    """
    ops = []

    if isinstance(e, Operator):
        ops.append(e)
    
    elif isinstance(e, Add):
        for arg in e.args:
            ops += extract_operator_products(arg, independent=independent)

    elif isinstance(e, Mul):
        c, o = split_coeff_operator(e)
        if o != 1:
            ops.append(o)
    else:
        print("Unrecongized type: %s" % type(e))
        
    no_ops = []
    for op in ops:
        no_op = normal_ordered_form(op.expand(), independent=independent)
        if isinstance(no_op, (Mul, Operator)):
            no_ops.append(no_op)
        elif isinstance(no_op, Add):
            for sub_no_op in extract_operator_products(no_op, independent=independent):
                no_ops.append(sub_no_op)
        else:
            raise ValueError("Unsupported type in loop over ops: %s: %s" %
                             (type(no_op), no_op))

    return list(set(no_ops))


def bch_expansion_auto(A, B, independent=False):

    # Use BCH expansion of order N
    N = 6
    
    c, _ = split_coeff_operator(A)

    if debug:
        print("A coefficient: ", c)

    e_bch = bch_expansion(A, B, N=N).doit(independent=independent)
    e = normal_ordered_form(e_bch.expand(), independent=independent)

    ops = extract_operator_products(e, independent=independent)

    # make sure that product operators comes first in the list
    ops = list(reversed(sorted(ops, key=lambda x: len(str(x)))))

    if debug:
        print("operators in expression: ", ops)

    e_collected = collect(e, ops)

    if c:
        return e_collected.subs({
                exp(c).series(c, n=N).removeO(): exp(c),
                exp(-c).series(-c, n=N).removeO(): exp(-c),
            })
    else:
        return e_collected

def unitary_transformation_auto(U, O, independent=False):
    """
    Perform a unitary transformation 

        O = U O U^\dagger

    and automatically try to identify series expansions in the resulting
    operator expression.
    """
    if not isinstance(U, exp):
        raise ValueError("U must be a unitary operator on the form U = exp(A)")    

    A = U.exp

    if debug:
        print("unitary_transformation_auto: using A = ", A)

    return bch_expansion_auto(A, O, independent=independent)


def hamiltonian_transformation_auto(U, H, independent=False):
    """
    Apply an unitary basis transformation to the Hamiltonian H:
    
        H = U H U^\dagger -i U d/dt(U^\dagger)
    
    """
    t = symbols("t")
    H_td = - I * U * diff(exp(-U.exp), t)
    #H_td = I * diff(U, t) * exp(- U.exp)  # hack: Dagger(U) = exp(-U.exp)
    H_st = unitary_transformation_auto(U, H, independent=independent)
    return H_st + H_td


def hamiltonian_transformation(H, UH, n):
    """
    Apply an unitary basis transformation to the Hamiltonian H:
    
    H = U H U^\dagger -i U d/dt(U^\dagger)
    
    U = exp(UH)
    
    """
    return - I * exp(UH) * diff(exp(-UH), t) + \
                normal_ordered_form(expand(bch_expansion(UH, H, n).doit(independent=True)), independent=True)


def drop_terms_containing(e, e_drops):
    """
    Drop terms contaning factors in the list e_drops
    """
    if isinstance(e, Add):
        # fix this
        e = Add(*(arg for arg in e.args if not any([e_drop in arg.args
                                                       for e_drop in e_drops])))
        e = Add(*(arg.subs({key: 0 for key in e_drops}) for arg in e.args))

    return e


def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e
    """
    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e
