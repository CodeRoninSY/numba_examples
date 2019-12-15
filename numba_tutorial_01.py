#!/usr/bin/env python
"""
numba_tutorial_01.py

<2019-12-14> CodeRoninSY

For details of Numba tutorial go to links:
https://www.youtube.com/watch?v=1AwG0T4gaO0
https://github.com/gforsyth/numba_tutorial_scipy2017

"""
import numpy
from time import sleep
import cProfile

def bad_call(dude):
    sleep(.5)

def worse_call(dude):
    sleep(1)

def sumulate(foo):
    if not isinstance(foo, int):
        return

    a = numpy.random.random((1000, 1000))
    a @ a

    ans = 0
    for i in range(foo):
        ans += i

    bad_call(ans)
    worse_call(ans)

    return ans


if __name__ == "__main__":
    b = sumulate(150)
    print(b)
    cProfile.run('sumulate(150)')
