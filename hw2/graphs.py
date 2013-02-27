import math
import matplotlib.pyplot as plt
import os
import pickle

def main():
    path = 'log/pass2/'
    dirs = os.listdir(path)
    dirs = filter(lambda x: '.log' in x, dirs)

    for log in dirs:
#    log = dirs[0]
        parsed = log.replace('_','').replace('.log','').split('-')
        netType = parsed[parsed.index('t')+1]
        learnRate = parsed[parsed.index('r')+1]
        
        if netType == 'simple':
            title = 'Simple Network, rate = %s' % learnRate
            filename = 'simple_%s' % learnRate
        elif netType == 'hidden':
            try:
                hidNodes = parsed[parsed.index('h')+1]
            except:
                hidNodes = '15'
            title = 'Hidden Network, %s nodes, rate = %s' % (hidNodes, learnRate)
            filename = 'hidden_%s_%s' % (learnRate, hidNodes)
        else:
            hidNodes = parsed[parsed.index('h')+1]
            p = parsed[parsed.index('p')+1]
            title = 'Custom Network, %s nodes, rate = %s, p = %s' % (hidNodes, learnRate, p)
            filename = 'custome_%s_%s_%s' % (learnRate, hidNodes, p)

        f = open(path+log, 'r')
        data = pickle.load(f)
        epochs = range(1,len(data)+1)
        train = [data[i][0] for i in xrange(len(data))]
        valid = [data[i][1] for i in xrange(len(data))]
        test = [data[i][2] for i in xrange(len(data))]
        plt.clf()
        plt.plot(epochs, train, '-r')
        plt.plot(epochs, valid, '-b')
        plt.plot(epochs, test, '-m')
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.legend(['training', 'testing', 'validation', 'Location', 'best'])
        plt.axis([0,100,0,1])
        plt.title(title)
        plt.savefig('graphs/'+filename+'.png')
        print filename

#    for i in xrange(len(data)):
            
    print "done!"

if __name__ == '__main__':
    main()
