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
#from IPython.display import display_latex
from IPython.display import Latex

class Covariance(Expr):
    """Covariance of two operators, expressed in terms of bracket <A, B>
    
    Parameters
    ==========
    
    A : Expr
        
    """
    is_commutative = True
    
    def __new__(cls, A, B):
        return Expr.__new__(cls, A, B)
    
    def _eval_expand_covariance(self, **hints):        
        A, B = self.args
        # <A + B, C> = <A, C> + <B, C>
        if isinstance(A, Add):
            return Add(*(Covariance(a, B).expand() for a in A.args))
        # <A, B + C> = <A, B> + <A, C>
        if isinstance(B, Add):
            return Add(*(Covariance(A, b).expand() for b in B.args))
        
        if isinstance(A, Mul):
            A = A.expand()            
            cA, ncA = A.args_cnc()
            return Mul(Mul(*cA), Covariance(Mul._from_args(ncA), B).expand())
        if isinstance(B, Mul):
            B = B.expand()            
            cB, ncB = B.args_cnc()
            return Mul(Mul(*cB), Covariance(A, Mul._from_args(ncB)).expand())        
        if isinstance(A, Integral):
            # <∫adx, B> ->  ∫<a, B>dx
            func, lims = A.function, A.limits
            new_args = [Covariance(func, B).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        if isinstance(B, Integral):
            # <A, ∫bdx> ->  ∫<A, b>dx
            func, lims = B.function, B.limits
            new_args = [Covariance(A, func).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        return self
    
    def doit(self, **hints):
        """ Evaluate covariance of two operators A and B """
        A = self.args[0]
        B = self.args[1]

        return Expectation(A*B) - Expectation(A) * Expectation(B)
    
    def _latex(self, printer, *args):
        return r"\left\langle %s, %s \right\rangle" % tuple([
            printer._print(arg, *args) for arg in self.args])

class Expectation(Expr):
    """Expectation Value of an operator, expressed in terms of bracket <A>
    
    Parameters
    ==========
    
    A : Expr
        The argument of the expectation value <A>
    """
    is_commutative = True
    
    def __new__(cls, A):
        return Expr.__new__(cls, A)
    
    def _eval_expand_expectation(self, **hints):
        A = self.args[0]
        if isinstance(A, Add):
        # <A + B> = <A> + <B>
            return Add(*(Expectation(a).expand() for a in A.args))

        if isinstance(A, Mul):
        # <c A> = c<A> where c is a commutative term
            A = A.expand()
            cA, ncA = A.args_cnc()
            return Mul(Mul(*cA), Expectation(Mul._from_args(ncA)).expand())
        
        if isinstance(A, Integral):
            # <∫adx> ->  ∫<a>dx
            func, lims = A.function, A.limits
            new_args = [Expectation(func).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        
        return self
    
    def eval_state(self, state):
        return qapply(Dagger(state) * self.args[0] * state, dagger=True).doit()
    
    def _latex(self, printer, *args):
        return r"\left\langle %s \right\rangle" % printer._print(self.args[0], *args)

def show_first_few_terms(e, n=10):
    if isinstance(e, Add):
        e_args_trunc = e.args[0:n]
        e = Add(*(e_args_trunc))
    
    return Latex("$" + latex(e).replace("dag", "dagger") + r"+ \dots$")

def exchange_integral_order(e):
    """
    exchanging integral order. Works in this way:
    ∫(∫ ... (∫(∫    dx_0)dx_1)... dx_n-1)dx_n -->  ∫(∫ ... (∫(∫  dx_1)dx_2)... dx_n)dx_0
    """
    if isinstance(e, Add):
        return Add(*[exchange_integral_order(arg) for arg in e.args])
    elif isinstance(e, Mul):
        return Mul(*[exchange_integral_order(arg) for arg in e.args])
    if isinstance(e, Integral):
        i = push_inwards(e)
        func, lims = i.function, i.limits
        if len(lims)>1:
            args = [func]
            for idx in range(1, len(lims)):
                args.append(lims[idx])
            args.append(lims[0])
            return(Integral(*args))
        else:
            return e
    else:
        return e
        
def pull_outwards(e, _n=0):
    """ 
    Trick to maximally pull out constant elements from the integrand,
    and expand terms inside the integrand.
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e
    if isinstance(e, Add):
        return Add(*[pull_outwards(arg, _n=_n+1) for arg in e.args]).expand()
    if isinstance(e, Mul):
        return Mul(*[pull_outwards(arg, _n=_n+1) for arg in e.args]).expand()
    elif isinstance(e, Integral):
        func = pull_outwards(e.function)
        dummy_var = e.variables
        if isinstance(func, Add):
            add_args = []
            for term in func.args:
                args = [term]
                for lim in e.limits:
                    args.append(lim)
                add_args.append(Integral(*args))
            e_new = Add(*add_args)
            return pull_outwards(e_new, _n=_n+1)
        elif isinstance(func, Mul):
            non_integral = Mul(*[arg for arg in func.args if not isinstance(arg, Integral)])
            integrals    = Mul(*[arg for arg in func.args if isinstance(arg, Integral)])

            const = Mul(*[arg for arg in non_integral.args if dummy_var[0] not in arg.free_symbols])
            nonconst = Mul(*[arg for arg in non_integral.args if dummy_var[0] in arg.free_symbols])
            if const==1:
                return e
            else:
                if len(dummy_var)==1:
                    return const * Integral(nonconst * integrals, e.limits[0])
                else:
                    args = [const * Integral(nonconst * integrals, e.limits[0])]
                    for lim in e.limits[1:]:
                        args.append(lim)
                    return pull_outwards(Integral(*args), _n=_n+1)
        else:
            return e
    else:
        return e
        
def push_inwards(e, _n=0):
    """
    Trick to push every factors into integrand
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e    
    if isinstance(e, Add):
        return Add(*[push_inwards(arg, _n=_n+1) for arg in e.args])
    elif isinstance(e, Mul):
        c = Mul(*[arg for arg in e.args if not isinstance(arg, Integral)])
        i_in = Mul(*[arg for arg in e.args if isinstance(arg, Integral)])
        if isinstance(i_in, Integral):
            func_in = i_in.function
            args = [c * func_in]
            for lim_in in i_in.limits:
                args.append(lim_in)
            return push_inwards(Integral(*args), _n=_n+1)
        else:
            return e
    elif isinstance(e, Integral):
        func = e.function
        new_func = push_inwards(func, _n=_n+1)

        args = [new_func]
        for lim in e.limits:
            args.append(lim)
        return Integral(*args)
    else:
        return e

class OperatorFunction(Operator):

    @property
    def operator(self):
        return self.args[0]

    @property
    def variable(self):
        return self.args[1]

    @property
    def free_symbols(self):
        return self.operator.free_symbols.union(self.variable.free_symbols)

    @classmethod
    def default_args(self):
        return (Operator("a"), Symbol("t"))

    def __call__(self, value):
        return OperatorFunction(self.operator, value)
    
    def __new__(cls, *args, **hints):
        if not len(args) in [2]:
            raise ValueError('2 parameters expected, got %s' % str(args))

        return Operator.__new__(cls, *args)

    def _eval_commutator_OperatorFunction(self, other, **hints):
        if self.operator.args[0] == other.operator.args[0]:
            if str(self.variable) == str(other.variable):
                return Commutator(self.operator, other.operator).doit()

        return None

    def _eval_adjoint(self):
        return OperatorFunction(Dagger(self.operator), self.variable)
    
    def _print_contents_latex(self, printer, *args):
        return r'{{%s}(%s)}' % (latex(self.operator), latex(self.variable))



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
                    exp(2*c).series(2*c, n=N).removeO(): exp(2*c), #c
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
                    sin(2*c).series(2*c, n=N).removeO(): sin(2*c),
                    cos(2*c).series(2*c, n=N).removeO(): cos(2*c),
                    sin(2*I*c).series(2*I*c, n=N).removeO(): sin(2*I*c),
                    sin(-2*I*c).series(-2*I*c, n=N).removeO(): sin(-2*I*c),
                    cos(2*I*c).series(2*I*c, n=N).removeO(): cos(2*I*c),
                    cos(-2*I*c).series(-2*I*c, n=N).removeO(): cos(-2*I*c),
                    #
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
        #e = Add(*(arg for arg in e.args if not any([e_drop in arg.args
        #                                               for e_drop in e_drops])))
                                                       
        new_args = []
        
        for term in e.args:
            
            keep = True
            for e_drop in e_drops:
                if e_drop in term.args:
                    keep = False
                    
                if isinstance(e_drop, Mul):
                    if all([(f in term.args) for f in e_drop.args]):
                        keep = False
            
            if keep:
        #        new_args.append(arg)
                new_args.append(term)
        e = Add(*new_args)
                                                       
        #e = Add(*(arg.subs({key: 0 for key in e_drops}) for arg in e.args))

    return e


def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e
    """
    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e
