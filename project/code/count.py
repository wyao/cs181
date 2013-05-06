import sys
import pickle

filename = sys.argv[1]

images = pickle.load(open(str(filename), "rb"))

print len(images)
