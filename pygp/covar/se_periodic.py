"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a :py:class:`pygp.covar.combinators.SumCF`
or :py:class:`pygp.covar.combinators.ProductCF` with the :py:class:`pygp.covar.noise.NoiseISOCF`, if you want noise to be modelled by this GP.
"""
import inspect

import scipy as SP
import logging as LG
from pygp.covar import CovarianceFunction
import dist
import pdb
import numpy

class Sqexp_Periodic_CF(CovarianceFunction):
    """
    Standart Squared Exponential Covariance function.

    **Parameters:**
    
    - dimension : int
        The dimension of this SE. For instance a 2D SE has
        hyperparameters like::
        
          covar_hyper = [Amplitude,1stD Length-Scale, 2ndD Length-Scale]

    - dimension_indices : [int]
        Optional: The indices of the n_dimensions in the input.
        For instance the n_dimensions of inputs are in 2nd and
        4th dimension dimension_indices would have to be [1,3].

    """   
    def __init__(self,*args,**kwargs):
        super(Sqexp_Periodic_CF, self).__init__(*args,**kwargs)
        self.n_hyperparameters = 2
        self.n_dimensions = 1

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = []
        names.append('SECF Amplitude')
        names.append('Negative Log Length Scale')
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.n_hyperparameters

    def K(self, theta, x1, x2=None, p=2):
        """
        See covPeriodic.m from GPML
        K(x1, x2) = A * exp( -nll*sin^2 (||x1-x2||/p) )
        """
        # print inspect.stack()[1][3], x1.shape, x2
        x1, x2 = self._filter_input_dimensions(x1, x2)
        A = SP.exp(theta[0])
        nll = SP.exp(theta[1])
        distance = dist.dist(x1, x2) / p
        sin_distance = numpy.sin(distance)
        rv = -nll * numpy.multiply(sin_distance, sin_distance)
        rv = A * SP.exp(rv)
        # This next line is kind of hacky. I get a (len, len, 1)
        # shape, so I just want to make it (len, len)
        # I should figure out the reason behind this in the first place, though
        rv.resize(rv.shape[:-1])
        return rv

#     def Kdiag(self,theta, x1):
#         """
#         Get diagonal of the (squared) covariance matrix.

#         **Parameters:**
#         See :py:class:`pygp.covar.CovarianceFunction`
#         """
#         #diagonal is independent of data
#         x1 = self._filter_x(x1)
#         V0 = SP.exp(2*theta[0])
#         return V0*SP.exp(0)*SP.ones([x1.shape[0]])
    
    def Kgrad_theta(self, theta, x1, i, x2=None, p=2):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1, x2 = self._filter_input_dimensions(x1, x2)
        # 2. exponentiate params:
        A = SP.exp(2*theta[0])
        nll  = SP.exp(theta[1:1+self.n_dimensions])
        # calculate sin^2 (|x1-x2|/p)
        sin_distance = numpy.sin(dist.dist(x1, x2) / p)
        sinsq = numpy.multiply(sin_distance, sin_distance)
        #3. calcualte withotu derivatives, need this anyway:
        derivative_A = SP.exp(-nll * sinsq)
        if i==0:
            return derivative_A
        else:
            return -A * sinsq * derivative_A

    
#     def Kgrad_x(self,theta,x1,x2,d):
#         """
#         The partial derivative of the covariance matrix with
#         respect to x, given hyperparameters `theta`.

#         **Parameters:**
#         See :py:class:`pygp.covar.CovarianceFunction`
#         """
#         # if we are not meant return zeros:
#         if(d not in self.dimension_indices):
#             return SP.zeros([x1.shape[0],x2.shape[0]])
#         rv = self.K(theta,x1,x2)
# #        #1. get inputs and dimension
#         x1, x2 = self._filter_input_dimensions(x1,x2)
#         d -= self.dimension_indices.min()
# #        #2. exponentialte parameters
# #        V0 = SP.exp(2*theta[0])
# #        L  = SP.exp(theta[1:1+self.n_dimensions])[d]
#         L2 = SP.exp(2*theta[1:1+self.n_dimensions])
# #        # get squared distance in right dimension:
# #        sqd = dist.sq_dist(x1[:,d]/L,x2[:,d]/L)
# #        #3. calculate the whole covariance matrix:
# #        rv = V0*SP.exp(-0.5*sqd)
#         #4. get non-squared distance in right dimesnion:
#         nsdist = -dist.dist(x1,x2)[:,:,d]/L2[d]
        
#         return rv * nsdist
    
#     def Kgrad_xdiag(self,theta,x1,d):
#         """"""
#         #digaonal derivative is zero because d/dx1 (x1-x2)^2 = 0
#         #because (x1^d-x1^d) = 0
#         RV = SP.zeros([x1.shape[0]])
#         return RV
