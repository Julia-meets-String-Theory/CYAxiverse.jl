using PyCall

"""
    __init__()
Here we initialise the CYTools functions 
(for further details and argument info see https://cytools.liammcallistergroup.com/docs/documentation/):

test_config() -- this function checks that the mosek_license file is found

f_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,
        n_dual_points,n_facets,limit,timeout,as_list,backend, dualize,favorable) -- 
            this function pulls polytopes from the online KS database, i.e. calls fetch_polytopes
            from CYTools

poly(points, backend) -- this function allows access to the PyObject Polytope
"""
function __init__()
    py"""
    from cytools import config
    import os
    config.mosek_license = os.path.join("/usr/users/mehta2")
    config.check_mosek_license()
    def test_config():
       return config.mosek_is_activated
    """

    py"""
    import numpy as np
    import scipy as sp
    from cytools import fetch_polytopes
    def f_polytopes(h11=None, h12=None, h13=None, h21=None, h22=None, h31=None,
                    chi=None, lattice=None, dim=4, n_points=None,
                    n_vertices=None, n_dual_points=None, n_facets=None,
                    limit=1000, timeout=60, as_list=False, backend=None,
                    dualize=False, favorable=None):
        return fetch_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,
        n_dual_points,n_facets,limit,timeout,as_list,backend, dualize,favorable)
    
    py"""
    import numpy as np
    import scipy as sp
    from cytools import Polytope
    def poly(points, backend=None):
        return Polytope(points,backend)
    """
end