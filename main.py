
import numpy as np
import random
import pylab 
import multiprocessing
import time
from pyrsistent import l
import timeit



def ifNone(a):
    if type(a) == type(None):
        return True
    else:
        return False


def maxPerArg(list1, list2):
    if len(list1) != len(list2):
        raise Exception("OOBA %s,%s" % (len(list1), len(list2)))
    ans = list1[:]
    for i in range(len(list1)):
        if list2[i]>list1[i]:
            ans[i] = list2[i]
    return ans


def minPerArg(list1, list2):
    if len(list1) != len(list2):
        raise Exception("OOBA %s,%s" % (len(list1), len(list2)))
    ans = list1[:]
    for i in range(len(list1)):
        if list2[i]<list1[i]:
            ans[i] = list2[i]
    return ans


class dataPoint:
    def __init__(self, d, id):
        self.data = d
        self.id = id
        self.ancestorsDistances = {}
        self.ancestorsDistancesList = []
        self.distanceFromDad = -1
        self.max = -1
        self.min = -1
        self.fars = [-1]
        self.nears = [-1]
    def __str__(self):
        return str(self.data)

random.seed(0)
def randomArray(size):
    ans = []
    for i in range(size):
        ans.append(random.random())
    return np.array(ans)

Objects = []
random.seed(1)
# f = open("data.list","r")
for i in range(500):
  tmp = dataPoint(randomArray(5), i)
  # tmp.data[0] = float(line.split(" ")[0])
  # tmp.data[1] = float(line.split(" ")[1])
  
  Objects.append(tmp)


class BallTreeNode:
  def __init__(self):
    self.myobject = None
    self.inner = None
    self.outer = None
    self.left = None
    self.right = None
    self.count = 0



def getData(myArray):
    
    myArray.sort(key=lambda x: x.distanceFromDad)
    theMax = myArray[-1].distanceFromDad
    theMin = myArray[0].distanceFromDad
    inner = myArray[int(len(myArray)/2)-1].distanceFromDad
    outer = myArray[int(len(myArray)/2)].distanceFromDad
    return theMax, theMin, inner, outer, myArray


counts = {}
def d(a, b, r=-1):
  global counts
  a, b = a.data, b.data
  counts[r]+=1
  return np.sqrt(np.sum((a-b)**2))


def ConstructBallTree(BT, localobjects):
  if len(localobjects)==0:
    return BT
  if len(localobjects)==1:
    BT.myobject = localobjects[0]
    BT.inner = 0
    BT.outer = 0
    BT.count = 1
    return BT
  selectedIndex = random.randint(0,len(localobjects)-1)
  localobjects[-1], localobjects[selectedIndex] = localobjects[selectedIndex], localobjects[-1] 
  BT.myobject = localobjects[-1]
  localobjects = localobjects[:-1]
  
  
  for i in range(len(localobjects)):
    localobjects[i].distanceFromDad=d(BT.myobject, localobjects[i])
    
    localobjects[i].ancestorsDistancesList.append(localobjects[i].distanceFromDad)
  
  
  BT.myobject.max, BT.myobject.min, BT.inner, BT.outer, sortedlocalobjects = getData(localobjects)

  
  

  L = []
  R = []
  for i in range(len(localobjects)):
    if localobjects[i].distanceFromDad<=BT.inner:
      L.append(localobjects[i])
    elif localobjects[i].distanceFromDad>=BT.outer:
      R.append(localobjects[i])
    else:
      print("where am i", BT.inner,BT.outer)

  BT.count = len(localobjects)

  if len(L) != 0:
    BT.left = ConstructBallTree(BallTreeNode(),L)
  if len(R) != 0:
    BT.right = ConstructBallTree(BallTreeNode(),R)
  
  

  return BT


def generateCMTData(BT):
    if ifNone(BT):
        return [], []
    Lnear, Lfar = generateCMTData(BT.left)
    Rnear, Rfar = generateCMTData(BT.right)
    Lnear, Lfar, Rnear, Rfar = Lnear[:-1], Lfar[:-1], Rnear[:-1], Rfar[:-1]
    
    if len(Lnear) != 0 and len(Rnear) != 0:
        
        BT.myobject.nears = minPerArg(minPerArg(Lnear, Rnear), BT.myobject.ancestorsDistancesList)
        BT.myobject.fars = maxPerArg(maxPerArg(Lfar, Rfar), BT.myobject.ancestorsDistancesList)
    elif len(Lnear) != 0:
        
        BT.myobject.nears = minPerArg(Lnear, BT.myobject.ancestorsDistancesList)
        BT.myobject.fars = maxPerArg(Lfar, BT.myobject.ancestorsDistancesList)
    elif len(Rnear) != 0:
        
        BT.myobject.nears = minPerArg(Rnear, BT.myobject.ancestorsDistancesList)
        BT.myobject.fars = maxPerArg(Rfar, BT.myobject.ancestorsDistancesList)
    elif len(Rnear) == 0 and len(Lnear) == 0:
    
        
        BT.myobject.nears = BT.myobject.ancestorsDistancesList
        BT.myobject.fars = BT.myobject.ancestorsDistancesList 
        
        
        
    
    
    returnNear = BT.myobject.nears
    returnFar = BT.myobject.fars

    return returnNear, returnFar
    


counts[-1] = 0
begin = time.time()
NewBT = ConstructBallTree(BallTreeNode(),Objects)
end = time.time()
print("Construction time:", (end-begin)*1000)

begin = time.time()
generateCMTData(NewBT)
end = time.time()
print("CMT Data time:", (end-begin)*1000)

def BaselineSearchRadius(BT, Object, r, ans):
  if BT == None:
    return
  dis = d(Object, BT.myobject, r)
  if dis <= r:
    ans.append(BT.myobject)
  

  if type(BT.myobject.max) != type(None):
    if dis > BT.myobject.max:
      if dis - BT.myobject.max > r: 
        return

  if type(BT.myobject.min) != type(None):
    if dis < BT.myobject.min:
      if BT.myobject.min - dis > r:
        return  
  
  
  
  if dis + r >= BT.outer:
    BaselineSearchRadius(BT.right, Object, r, ans)
  if BT.inner >= dis - r:                              
    BaselineSearchRadius(BT.left, Object, r, ans)


def maxPD(stack, node): 
    nears = node.myobject.nears
    fars = node.myobject.fars
    if len(stack) != len(fars):
        
        raise Exception("stack and fars not match %s, %s" %(len(stack), len(fars)))
    maxPDV = 0
    for i in range(len(stack)):
        
        if fars[i] < stack[i]:
            
            maxPDV = max(maxPDV, stack[i] - fars[i]) 
        elif nears[i] > stack[i]:
            maxPDV = max(maxPDV, nears[i] - stack[i])
    # print(maxPDV)
    return maxPDV
            


def CMTSearchRadius(BT, Object, r, ans):
  global stack
  if BT == None:
    return

  maxPDV = maxPD(stack, BT) 
      
  if maxPDV > r:
    return
    
  dis = d(Object, BT.myobject, r)
    
  if dis <= r:
    ans.append(BT.myobject)  


  

  if type(BT.myobject.max) != type(None):
    if dis > BT.myobject.max:
      if dis - BT.myobject.max > r: 
        return

  if type(BT.myobject.min) != type(None):
    if dis < BT.myobject.min:
      if BT.myobject.min - dis > r:
        return  
  
  
  
  stack.append(dis)
  if dis + r >= BT.outer:
    CMTSearchRadius(BT.right, Object, r, ans)
  if BT.inner >= dis - r:                              
    CMTSearchRadius(BT.left, Object, r, ans)
  stack.pop()


stack =[]
def BaselineTest(r):
  ans = []
  counts[r] = 0
  tp = 10
  stack = []
  for _ in range(tp):
    BaselineSearchRadius(NewBT, random.choice(Objects), r, ans)
  
  
  found = len(ans)
  
  return counts[r]/tp


def CMTTest(r):
    ans = []
    counts[r] = 0
    tp = 10
    stack = []
    for _ in range(tp):
        CMTSearchRadius(NewBT, random.choice(Objects), r, ans)
    
    
    found = len(ans)
    
    return counts[r]/tp


def compare(r):
    ans = []
    BTans = []
    counts[r] = 0
    CMTSearchRadius(NewBT, Objects[1], r, ans)
    
    
    found = len(ans)
    for i in Objects:
        if d(Objects[1],i)<=r:
            BTans.append(i)
    if len(BTans) != found:
        print(len(BTans), found)
        return len(BTans) - found
    else:
        return 0


myRange = np.arange(0,1.1,0.1)

print("Model Name, Language, r, nFound, Call Counts, Time")


def BFSearchRadius(q, r, ans):
  for i in Objects:
    if d(q,i)<=r:
      ans.append(i)

baseLineAnswer = []
baselineFound = []
for i in myRange:
    counts[i] = 0
    RT = 0
    for _ in range(20): 
      RT += timeit.timeit(lambda: BaselineSearchRadius(NewBT, Objects[_*10], i, baselineFound), number=1)
    print("Baseline Real, Python, %s, %s, %s, %s" % (i, len(baselineFound)/20, counts[i]/20, RT/20*1000))
    # print("Baseline time when r=%s on python:" %i , (end-begin)*1000)
    baseLineAnswer.append(counts[i])


CMTAnswer = []
CMTFound = []
for i in myRange:
    counts[i] = 0
    RT = 0
    for _ in range(20): 
      RT += timeit.timeit(lambda: CMTSearchRadius(NewBT, Objects[_*10], i, CMTFound), number=1)
    print("CMT Real, Python, %s, %s, %s, %s" % (i, len(CMTFound)/20, counts[i]/20, RT/20*1000))
    CMTAnswer.append(counts[i])

BFans = []
BFFound = []
for i in myRange:
    counts[i] = 0
    RT = 0
    for _ in range(20): 
      RT += timeit.timeit(lambda: BFSearchRadius(Objects[_*10], i, BFFound), number=1)
    BFans.append(counts[i])

import pylab
pylab.plot(baseLineAnswer)
pylab.plot(CMTAnswer)
pylab.savefig("ans.png")
print(f"SUM: {np.sum(len (BFFound))-len(np.array(CMTFound))}")

