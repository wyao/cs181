l = []

f = open("typescript")
for line in f:
    if line.find("Player 1") > -1:
        l.append(1)
    elif line.find("Player 2") > -1:
        l.append(2)

x = 100
win = 0.
for i in xrange(1, len(l)+1):
    if l[i] == 1:
        win += 1
    if i % x == 0:
        print int(i / x), win / x
        win = 0.
        