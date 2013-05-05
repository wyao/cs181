import pickle
from optparse import OptionParser
import py.test
import common
import random
from ann.ann import *
from ann.ann_impl import *

network = None
data = []

# Label 0: Poisonous, Label 1: Nutritious
def train(opts):
    global network, data
    p_count = 0
    # Load samples
    f = open(opts.f, "r")
    data = pickle.load(f)
    f.close()
    # Consolidate data
    h = {}
    for d in data:
        img = tuple(d[1])
        if img in h:
            p,n = h[img]
            if d[0]:
                n += 1
            else:
                p += 1
            h[img] = p,n
        else:
            if d[0]:
                h[img] = 0,1
            else:
                h[img] = 1,0
    # Initialize network
    network = HiddenNetwork(opts.hidden)
    network.FeedForwardFn = FeedForward
    network.TrainFn = Train
    network.InitializeWeights("")
    # Train
    for i in xrange(opts.e):
        print "Epoch:", i
        for k,v in h.iteritems():
            p,n = v
            if not (p > 0 and n > 0):
                if n > 0 or p_count < 2000 * opts.e:
                    if p > 0:
                        p_count += 1
                    target = [0., 1.] if v[1] > 0 else [1., 0.]
                    Backprop(network.network, k, target, opts.l)
        # for d in data: # TODO set aside validation set
        #     target = [0., 1.] if d[0] else [1., 0.]
        #     Backprop(network.network, d[1], target, opts.l)

def test_performance(opts):
    count = 0.
    correct = 0.
    for d in data:
        print network.Classify(d[1])
        if network.Classify(d[1]) == d[0]:
            correct += 1
        count += 1
    print correct/count

def main():
    parser = OptionParser()
    parser.add_option("--hidden", action="store", default=10, type=int)
    parser.add_option("-l", action="store", default=.1, type=float)
    parser.add_option("-f", action="store", default=None, type="string")
    parser.add_option("-e", action="store", default=10, type=int)
    (opts,args) = parser.parse_args()
    assert(opts.f != None)
    train(opts)
    test_performance(opts)

if __name__ == "__main__":
    main()
