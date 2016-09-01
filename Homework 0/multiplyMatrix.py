Python 2.7.12 (v2.7.12:d33e0cf91556, Jun 27 2016, 15:19:22) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy
>>> a = numpy.matrix([[1, 4, 3],[2, -1, 3]])
>>> b = numpy.matrix([[-2, 0, 5],[0, -1, 4]])
>>> a*b

Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    a*b
  File "C:\Python27\lib\site-packages\numpy\matrixlib\defmatrix.py", line 343, in __mul__
    return N.dot(self, asmatrix(other))
ValueError: shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)
>>> 
