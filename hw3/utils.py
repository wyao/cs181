import itertools

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best


def squareDistance(xs, ys):
    """ Computes the square distance of two vectors in n dimensions
    >>> squareDistance([1,2,3], [2,3,4])
    """
    res = sum( [(x - y)**2 for x,y in zip(xs,ys)] )
    return(res)

def cmin(c1, c2, d):
    """ computes the min distance between two clusters `c1` and `c2`, using
    `d` as a reference measure
    example:
    c1 = [ [1,2], [2,3] ]
    c2 = [ [2,3], [6,7] ]
    ccent(c1, c2, squareDistance)
    """
    min_dist = d(c1[0], c2[0])
    for x in c1:
        for y in c2:
            tmp_dist = d(x,y)
            if tmp_dist < min_dist:
                min_dist = tmp_dist
    return(min_dist)


def cmax(c1, c2, d):
    """ computes the max distance between two clusters `c1`and `c2`, using
    `d` as a reference measure
    """
    max_dist = d(c1[0], c2[0])
    for x in c1:
        for y in c2:
            tmp_dist = d(x,y)
            if tmp_dist > max_dist:
                max_dist = tmp_dist
    return(max_dist)


def cmean(c1, c2, d):
    """ computes the mean distance between two clusters `c1`and `c2`, using
    `d` as a reference measure
    """
    scale = float(1) / float(len(c1) * len(c2))
    res = scale * sum(map(lambda x: d(x[0], x[1]), itertools.product(c1,c2)))
    return(res)


def ccent(c1, c2, d):
    """ computes the mean distance between two clusters `c1`and `c2`, using
    `d` as a reference measure
    """

    def vsum(xs):
        """ computes the vectorialized sum of a list of vectors
        example:
        vsum([ [1,2], [3,4], [5,6] ]) => [9, 12]
        """
        new_x = [0 for i in range(len(xs[0]))]
        for x in xs:
            for i in range(len(new_x)):
                new_x[i] = new_x[i] + x[i]
        return(new_x)
    
    nx = map(lambda x: x * float(1) / float(len(c1)),  vsum(c1))
    ny = map(lambda x: x * float(1) / float(len(c2)),  vsum(c2))

    return d(nx, ny)

attributes = [
    {'attribute':"age: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Private", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Self-emp-not-inc", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Self-emp-inc", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Federal-gov", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Local-gov", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-State-gov", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Without-pay", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"workclass-Never-worked.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"fnlwgt: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"education-num: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Married-civ-spouse", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Divorced", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Never-married", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Separated", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Widowed", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Married-spouse-absent", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"marital-status-Married-AF-spouse.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Tech-support", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Craft-repair", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Other-service", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Sales", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Exec-managerial", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Prof-specialty", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Handlers-cleaners", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Machine-op-inspct", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Adm-clerical", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Farming-fishing", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Transport-moving", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Priv-house-serv", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Protective-serv", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"occupation-Armed-Forces.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Wife", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Own-child", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Husband", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Not-in-family", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Other-relative", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"relationship-Unmarried.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"race-White", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"race-Asian-Pac-Islander", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"race-Amer-Indian-Eskimo", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"race-Other", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"race-Black.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"sex: 0 = Female, 1 = Male.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"capital-gain: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"capital-loss: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"hours-per-week: continuous.", 'isContinuous':True, 'min':None, 'max':None, 'intervals':None},
    {'attribute':"income: 0 = >50K, 1 = <=50K.", 'isContinuous':False, 'min':None, 'max':None, 'intervals':None}
]

import math

def assignClass(x,d,attr):
    if attr['isContinuous']:
        denominator = (attr['max'] - attr['min']) / attr['intervals']
        return int(math.floor( (x[d] - attr['min']) / denominator ))
    else:
        return 1 if x[d] > 0 else 0
