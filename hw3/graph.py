filename = 'test'
plt.clf()
plt.plot(xValues, yValues, '-r')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['legendHere'])
plt.axis([0,100,0,1])
plt.title(title)
plt.savefig('graphs/'+filename+'.png')
