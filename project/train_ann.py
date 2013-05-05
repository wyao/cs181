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
    f = open(opts.train_file, "r")
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
    network = SimpleNetwork()#HiddenNetwork(opts.hidden)
    network.FeedForwardFn = FeedForward
    network.TrainFn = Train
    network.InitializeWeights("")
    # Train
    for i in xrange(opts.e):
        print "Epoch:", i+1
        for k,v in h.iteritems():
            p,n = v
            # Filter out conflicting data
            if not (p > 0 and n > 0):
                # if n > 0 or p_count < 4000 * opts.e:
                #     if p > 0:
                #         p_count += 1
                    target = [0., 1.] if v[1] > 0 else [1., 0.]
                    Backprop(network.network, k, target, opts.l)

def test_performance(opts):
    count = 0.
    correct = 0.
    for d in data:
        # print network.Classify(d[1])
        if network.Classify(d[1]) == d[0]:
            correct += 1
            if network.Classify(d[1]) > 0:
                print "N+"
            # else:
            #     print "P+"
        else:
            if network.Classify(d[1]) > 0:
                print "N-"
            # else:
            #     print "P-"
        count += 1
    # Now for validation set
    if opts.v_file != None:
        print "Validating..."
        count_ = 0.
        correct_ = 0.
        n_correct = 0.
        n_count = 0.
        f = open(opts.v_file, "r")
        validation = pickle.load(f)
        f.close()
        for d in validation:
            if network.Classify(d[1]) == d[0]:
                correct_ += 1
                if network.Classify(d[1]) > 0:
                    print "N+"
                    n_correct += 1
                    n_count += 1
                # else:
                #     print "P+"
            else:
                if network.Classify(d[1]) > 0:
                    print "N-"
                    n_count += 1
                # else:
                #     print "P-"
            count_ += 1
    print "Test:", correct/count
    if opts.v_file != None:
        print "Validation:", correct_ / count_
        print "Performance on N:", n_correct / n_count

def main():
    parser = OptionParser()
    parser.add_option("--hidden", action="store", default=10, type=int)
    parser.add_option("-l", action="store", default=.1, type=float)
    parser.add_option("--train_file", action="store", default=None, type="string")
    parser.add_option("--v_file", action="store", default=None, type="string")
    parser.add_option("--out_file", action="store", default=None, type="string")
    parser.add_option("-e", action="store", default=10, type=int)
    parser.add_option("-v", action="store", default=100, type=int)
    (opts,args) = parser.parse_args()
    assert(opts.train_file != None)
    train(opts)
    test_performance(opts)
    if opts.out_file != None:
        print "Exporting"
        network.ExportWeights(opts.out_file)

if __name__ == "__main__":
    main()
