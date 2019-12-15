#!/usr/bin/env python
"""
numba_direct_sum_04.py

<2019-12-15> CodeRoninSY

N-Body problems and usage of Numba JIT

Numba doesn't support jitting native Python classes.
Solution: numpy custom dtypes

https://github.com/gforsyth/numba_tutorial_scipy2017

"""
import numpy
from line_profiler import LineProfiler
from timer import timeit


class Point():
    """
    Arguments:
    domain: the domain of random coordinates x,y,z, default=1.0
    Attributes:
    x,y,z: coordinates of the point
    """
    def __init__(self, domain=1.0):
        self.x = domain * numpy.random.random()
        self.y = domain * numpy.random.random()
        self.z = domain * numpy.random.random()

    def distance(self, other):
        return ((self.x - other.x) ** 2 +
                (self.y - other.y) ** 2 +
                (self.z - other.z) ** 2) ** .5


class Particle(Point):
    """
    Attributes:
        m: mass of the particle
        phi: the potential of the particle
    """

    def __init__(self, domain=1.0, m=1.0):
        Point.__init__(self, domain)
        self.m = m
        self.phi = 0.

# @timeit
def direct_sum(particles):
    """
    Calculate the potential at each particle
    using direct summation method.

    Arguments:
        particles: the list of particles

    """
    for i, target in enumerate(particles):
        for source in (particles[:i] + particles[i+1:]):
            r = target.distance(source)
            target.phi += source.m / r


if __name__ == "__main__":
    n = 1000
    particles = [Particle(m=1 / n) for i in range(n)]
    direct_sum(particles)

    # line_profiler for direct_sum
    lp = LineProfiler()
    lp.add_function(timeit)
    lp_wrapper = lp(direct_sum)
    lp_wrapper(particles)
    lp.print_stats()

    particle_dtype = numpy.dtype({'names': ['x', 'y', 'z', 'm', 'phi'],
                              'formats': [numpy.double,
                                          numpy.double,
                                          numpy.double,
                                          numpy.double,
                                          numpy.double]})

    myarray = numpy.ones(3, dtype=particle_dtype)
    print(myarray)
