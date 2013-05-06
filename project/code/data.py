import pickle
import py.test
import matplotlib.pyplot as plt
import numpy as np, numpy.random
import scipy

filename = '1000-spiral.txt' #100-delimited.txt'
output = '100-games.txt'
f = open(filename)
data = pickle.load(f)

poisonousX = []
poisonousY = []
nutritiousX = []
nutritiousY = []

allX = []
allY = []

game_num = 10

plants = {}

for i in data:
	num,status,image,x,y = i
	if num in plants:
		plants[num] += 1
	else:
		plants[num] = 1
	# if num == 0:
	if status == 0:
		poisonousX.append(x)
		poisonousY.append(y)
	else:
		nutritiousX.append(x)
		nutritiousY.append(y)
	allX.append(x)
	allY.append(y)

# plt.clf()
# plt.plot(poisonousX, poisonousY, 'xr')
# plt.plot(nutritiousX, nutritiousY, 'xb')
# # plt.plot(allX, allY, '-k')
# plt.show()
targetX = nutritiousX
targetY = nutritiousY
#histogram definition
xyrange = [[-20,20],[-20,20]] # data range
bins = [40,40] # number of bins
thresh = 3  #density threshold

#data definition
N = 1e5
xdat, ydat = np.array(targetX), np.array(targetY) #np.random.normal(size=N), np.random.normal(1, 0.6, size=N)#allX, allY

# py.test.set_trace()
# histogram the data
hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
posx = np.digitize(xdat, locx)
posy = np.digitize(ydat, locy)

#select points within the histogram
ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
# py.test.set_trace()
xdat1 = xdat[ind][hhsub < thresh] # low density points
ydat1 = ydat[ind][hhsub < thresh]
hh[hh < thresh] = np.nan # fill the areas with low density by NaNs

plt.clf()
plt.imshow(hh.T,cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none')
plt.colorbar()   
plt.plot(xdat1, ydat1, '.')
plt.show()

print plants

# hashtable = {}	
# for i in data:
# 	if i[1] in hashtable:
# 		if i[0] == 0:
# 			hashtable[i[1]][0] = hashtable[i[1]][0]+1
# 		else:
# 			hashtable[i[1]][1] = hashtable[i[1]][1]+1
# 	else:
# 		hashtable[i[1]] = 

# newData = []
# thisGame = []
# for i in data:
# 	if i[0] != 2:
# 		thisGame.append(i)
# 	else:
# 		newData.append(thisGame)
# 		thisGame = []

# g = open(output, 'wb')
# pickle.dump(newData, g)

# games = []
# for i in newData:
# 	games.append([x[0] for x in i])

# for i in games:
# 	if 1 in i:
# 		first = i.index(1)
# 	else:
# 		first = -1
# 	total = len(i)
# 	# print float(first)/total, float(sum(i))/total, total

# near = 0
# count = 0

# test = [[0,0,1,1,1,0,0,0,0,1,0,0,0],[0,0,1,1,0,1]]

# for i in games:
# 	prev = -1
# 	twoprev = -1
# 	for j in i:
# 		if j == 1 and prev == 1 and twoprev == 0:
# 			near += 2
# 		if j == 1 and prev == 1 and twoprev == 1:
# 		 	near += 1
# 		if j == 1 and prev == 0 and twoprev == 1:
# 		 	near += 1
# 		if j == 1:
# 			count += 1
# 		twoprev = prev
# 		prev = j
# # 		print (near, count, prev)
			
# # print count
# # print near

# # for i in games:
# # 	print i

# # py.test.set_trace()