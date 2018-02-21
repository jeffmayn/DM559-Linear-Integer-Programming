def constructSet():
    print '\nConstruct in Python the set S = {x E R | x >= 0 A x mod 3 = 1}'
    print "--------------------------------------------------------------------"
    S = {x for x in range(50) if x % 3 == 1}
    print(S)

def comprehension():
    print "\nUsing comprehension lists make a list for {(i, j) | i E {1, 2, 3, 4}, j E {5, 7, 9}}"
    print "----------------------------------------------------------------------------------------------"
    val = [(i+1,j) for i in range(4) for j in [5,7,9]]
    print(val)

def inverse():
    print "\nCalculate the inverse of a function or the index function for an invertible"
    print "function (ie, bijective = injective + surjective) given in form of a dictionary."
    print "---------------------------------------------------------------------------------"
    {d[k]:k for k in d}
    {v:k for k in d.keys() for v in d.values}
    {v:k for k,v in d.items()}

constructSet()
print "\n"
comprehension()
print "\n"
inverse()
print "\n\n"
