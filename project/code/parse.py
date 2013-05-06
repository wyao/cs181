import sys

# python parse.py q_data/2000.p; python parse.py q_data/5000.p; python parse.py q_data/10000.p; python parse.py q_data/20000.p
# 1 0.245454545455
# 1 0.177777777778
# 1 0.30101010101
# 1 0.384848484848

l = []

f = open(sys.argv[1])
for line in f:
    if line.find("Player 1") > -1:
        l.append(1)
    elif line.find("Player 2") > -1:
        l.append(2)

x = 990
win = 0.
for i in xrange(1, len(l)+1):
    if l[i-1] == 1:
        win += 1
    if i % x == 0:
        print int(i / x), win / x
        win = 0.
        