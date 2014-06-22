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
def qsimplify(e_orig, _n=0):
    
    if _n > 15:
        warnings.warn("Too high level or recursion, aborting")
        return e_orig

    e = normal_ordered_form(e_orig)
    
    if isinstance(e, Add):
        return Add(*(qsimplify(arg, _n=_n+1) for arg in e.args))

    
    elif isinstance(e, Mul):
        args1 = tuple(arg for arg in e.args if arg.is_commutative)
        args2 = tuple(arg for arg in e.args if not arg.is_commutative)
        #x = 1
        #for y in args2:
        #    x = x * y

        x = 1
        for y in reversed(args2):
            x = y * x
            
        if isinstance(x, Mul):
            args2 = x.args
            x = 1
            for y in args2:
                x = x * y
            
            
        e_new = simplify(Mul(*args1)) * x

        if e_new == e:
            return e
        else:
            return qsimplify(e_new.expand(), _n=_n+1)
   

    if e == e_orig:
        return e
    else:
        return qsimplify(e, _n=_n+1).expand()

def recursive_commutator(a, b, n=1):
    return Commutator(a, b) if n == 1 else Commutator(a, recursive_commutator(a, b, n-1))


def _bch_expansion(A, B, N=10):
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

                if c and c != 1:
                    c_args.append(c ** arg.exp)
                if o and o != 1:
                    o_args.append(o ** arg.exp)
            else:
                c_args.append(arg)
                
        return Mul(*c_args), Mul(*o_args)

    if isinstance(e, Add):
        # XXX: fix this  -> return to lists
        return split_coeff_operator(e.args[0])

    if debug:
        print("Warning: Unrecognized type of e: %s" % type(e))

    return None, None


def extract_operators(e, independent=False):
    """
    Return a list of unique normal-ordered quantum operator products in the
    expression e.
    """
    ops = []

    if isinstance(e, Operator):
        ops.append(e)
    
    elif isinstance(e, Add):
        for arg in e.args:
            ops += extract_operators(arg, independent=independent)

    elif isinstance(e, Mul):
        for arg in e.args:
            ops += extract_operators(arg, independent=independent)
    else:
        if debug:
            print("Unrecongized type: %s: %s" % (type(e), str(e)))
        
    return list(set(ops))


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
        if debug:
            print("Unrecongized type: %s: %s" % (type(e), str(e)))
        
    no_ops = []
    for op in ops:
        no_op = normal_ordered_form(op.expand(), independent=independent)
        if isinstance(no_op, (Mul, Operator, Pow)):
            no_ops.append(no_op)
        elif isinstance(no_op, Add):
            for sub_no_op in extract_operator_products(no_op, independent=independent):
                no_ops.append(sub_no_op)
        else:
            raise ValueError("Unsupported type in loop over ops: %s: %s" %
                             (type(no_op), no_op))

    return list(set(no_ops))


def bch_expansion(A, B, N=6, collect_operators=None, independent=False,
                  expansion_search=True):

    # Use BCH expansion of order N

    if debug:
        print("bch_expansion: ", A, B)

    
    c, _ = split_coeff_operator(A)

    if debug:
        print("A coefficient: ", c)

    if debug:
        print("bch_expansion: ")

    e_bch = _bch_expansion(A, B, N=N).doit(independent=independent)

    if debug:
        print("simplify: ")

    e = qsimplify(normal_ordered_form(e_bch.expand(), 
                                     recursive_limit=25,
                                     independent=independent).expand())

    if debug:
        print("extract operators: ")

    ops = extract_operator_products(e, independent=independent)

    # make sure that product operators comes first in the list
    ops = list(reversed(sorted(ops, key=lambda x: len(str(x)))))

    if debug:
        print("operators in expression: ", ops)

    if collect_operators:
        e_collected = collect(e, collect_operators)        
    else:
        e_collected = collect(e, ops)

    if debug:
        print("search for series expansions: ", expansion_search)

    try:
        if expansion_search and c:
            c_fs = list(c.free_symbols)[0]
            if debug:
                print("free symbols in c: ", c_fs)
            return qsimplify(e_collected.subs({
                    exp(c).series(c, n=N).removeO(): exp(c), #c
                    exp(-c).series(-c, n=N).removeO(): exp(-c), #-c
                    exp(2*c).series(c, n=N).removeO(): exp(2*c), #c
                    exp(-2*c).series(-2*c, n=N).removeO(): exp(-2*c), #-c, list(c.free_symbols)[0]
                    #
                    cosh(c).series(c, n=N).removeO(): cosh(c),
                    sinh(c).series(c, n=N).removeO(): sinh(c),
                    sinh(2*c).series(2 * c, n=N).removeO(): sinh(2*c),
                    cosh(2*c).series(2 * c, n=N).removeO(): cosh(2*c),
                    sinh(4*c).series(4 * c, n=N).removeO(): sinh(4*c),
                    cosh(4*c).series(4 * c, n=N).removeO(): cosh(4*c),
                    #
                    sin(c).series(c, n=N).removeO(): sin(c),
                    cos(c).series(c, n=N).removeO(): cos(c),
                    sin(2*c).series(c, n=N).removeO(): sin(2*c),
                    cos(2*c).series(c, n=N).removeO(): cos(2*c),
                    sin(c_fs).series(c_fs, n=N).removeO(): sin(c_fs),
                    cos(c_fs).series(c_fs, n=N).removeO(): cos(c_fs),
                    (sin(c_fs)/2).series(c_fs, n=N).removeO(): sin(c_fs)/2,
                    (cos(c_fs)/2).series(c_fs, n=N).removeO(): cos(c_fs)/2,
                    #sin(2*c_fs).series(c_fs, n=N).removeO(): sin(2*c_fs),
                    #cos(2*c_fs).series(c_fs, n=N).removeO(): cos(2*c_fs),
                    #sin(2 * c_fs).series(2 * c_fs, n=N).removeO(): sin(2 * c_fs),
                    #cos(2 * c_fs).series(2 * c_fs, n=N).removeO(): cos(2 * c_fs),
                    #(sin(c_fs)/2).series(c_fs, n=N).removeO(): sin(c_fs)/2,
                    #(cos(c_fs)/2).series(c_fs, n=N).removeO(): cos(c_fs)/2,
                }))  
        else:
            return e_collected
    except Exception as e:
        print("Failed to identify series expansions: " + str(e))
        return e_collected


def subs_single(O, subs_map):

    if isinstance(O, Operator):
        if O in subs_map:
            return subs_map[O]
        else:
            print("warning: unresolved operator: ", O)
            return O
    elif isinstance(O, Add):
        new_args = []
        for arg in O.args:
            new_args.append(subs_single(arg, subs_map))
        return Add(*new_args)

    elif isinstance(O, Mul):
        new_args = []
        for arg in O.args:
            new_args.append(subs_single(arg, subs_map))
        return Mul(*new_args)

    elif isinstance(O, Pow):
        return Pow(subs_single(O.base, subs_map), O.exp)

    else:
        return O


def unitary_transformation(U, O, N=6, collect_operators=None,
                           independent=False, allinone=False,
                           expansion_search=True):
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
        print("unitary_transformation: using A = ", A)


    if allinone:
        return bch_expansion(A, O, N=N, collect_operators=collect_operators,
                             independent=independent,
                             expansion_search=expansion_search)
    else:
        ops = extract_operators(O.expand())
        ops_subs = {op: bch_expansion(A, op, N=N,
                                      collect_operators=collect_operators,
                                      independent=independent,
                                       expansion_search=expansion_search)
                    for op in ops}

        #return O.subs(ops_subs)
        return subs_single(O, ops_subs)


def hamiltonian_transformation(U, H, N=6, collect_operators=None, independent=False,
                               expansion_search=True):
    """
    Apply an unitary basis transformation to the Hamiltonian H:
    
        H = U H U^\dagger -i U d/dt(U^\dagger)
    
    """
    t = [s for s in U.exp.free_symbols if str(s) == 't']
    if t:
        t = t[0]
        H_td = - I * U * diff(exp(-U.exp), t)
    else:
        H_td = 0
        
    #H_td = I * diff(U, t) * exp(- U.exp)  # hack: Dagger(U) = exp(-U.exp)
    H_st = unitary_transformation(U, H, N=N, collect_operators=collect_operators,
                                  independent=independent, expansion_search=expansion_search)
    return H_st + H_td


def drop_terms_containing(e, e_drops):
    """
    Drop terms contaning factors in the list e_drops
    """
    if isinstance(e, Add):
        # fix this
        e = Add(*(arg for arg in e.args if not any([e_drop in arg.args
                                                       for e_drop in e_drops])))
        #e = Add(*(arg.subs({key: 0 for key in e_drops}) for arg in e.args))

    return e


def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e
    """
    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e
