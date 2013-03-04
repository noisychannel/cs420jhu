'''
Created on Feb 15, 2013

@author: "Gaurav Kumar"
'''

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import collections


def getSerialData(logFileLocation):
    
    serialXY = []
    serialYX = []
    
    logFile = open(logFileLocation)
    for line in logFile:
        runtime = re.findall('\d+',line)
        if "Serial YX" in line:
            serialYX.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
            
        if "Serial XY" in line:
            serialXY.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
        
    return np.mean(serialXY), np.mean(serialYX)
    
    
def getParallelData(logFileLocation):    
    
    parallelYX = []
    parallelXY = []
    parallelCoal = []
    mergedDualParallel = []
    dualParallel = []
    
    parallelYXMeans = {1:0,2:0,4:0,8:0,16:0,32:0}
    parallelXYMeans = {1:0,2:0,4:0,8:0,16:0,32:0}
    parallelCoalMeans = {1:0,2:0,4:0,8:0,16:0,32:0}
    mergedDualParallelMeans = {1:0,2:0,4:0,8:0,16:0,32:0}
    dualParallelMeans = {1:0,2:0,4:0,8:0,16:0,32:0}
    
    logFile = open(logFileLocation)
    for line in logFile:
        runtime = re.findall('\d+',line)
            
        if "Parallel YX" in line:
            parallelYX.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
            
        elif "Parallel XY" in line:
            parallelXY.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
            
        elif "Parallel coalesced" in line:
            parallelCoal.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
            
        elif "Merged Dual Parallel" in line:
            mergedDualParallel.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
            
        else:
            if "Iteration" not in line:
                if "Serial YX" not in line:
                    if "Serial XY" not in line:
                        if "Threads" not in line:
                            dualParallel.append(float(float(runtime[0]) + float(float(runtime[1])/1000000.0)))
                            
    for i in range(6):
        totalParallelYX = 0.0
        totalParallelXY = 0.0
        totalParallelCoal = 0.0
        totalMergedDualParallel = 0.0
        totalDualParallel = 0.0
        for j in range(0,120,6):
            totalParallelYX = totalParallelYX + parallelYX[i+j]
            totalParallelXY = totalParallelXY + parallelXY[i+j]
            totalParallelCoal = totalParallelCoal + parallelCoal[i+j]
            totalMergedDualParallel = totalMergedDualParallel + mergedDualParallel[i+j]
            totalDualParallel = totalDualParallel + dualParallel[i+j]
        parallelYXMeans[pow(2,i)] = totalParallelYX/20.0
        parallelXYMeans[pow(2,i)] = totalParallelXY/20.0
        parallelCoalMeans[pow(2,i)] = totalParallelCoal/20.0
        mergedDualParallelMeans[pow(2,i)] = totalMergedDualParallel/20.0
        dualParallelMeans[pow(2,i)] = totalDualParallel/20.0

    return parallelYXMeans, parallelXYMeans, parallelCoalMeans, mergedDualParallelMeans, dualParallelMeans


if __name__ == '__main__':

    parallelXY = []
    parallelYX = []
    parallelCoal = []
    dualParallel = []
    mergedDualParallel = []
    
#    logFileLocation = "../logs/halfwidth-1.log"
#    serialXYMean1 , serialYXMean1 = getSerialData(logFileLocation)
#    print 'For halfwidth 1 : Serial XY Mean = ' + str(serialXYMean1)  
#    print 'For halfwidth 1 : Serial YX Mean = ' + str(serialYXMean1)
#    
#    logFileLocation = "../logs/halfwidth-2.log"
#    serialXYMean2 , serialYXMean2 = getSerialData(logFileLocation)
#    print 'For halfwidth 2 : Serial XY Mean = ' + str(serialXYMean2)  
#    print 'For halfwidth 2 : Serial YX Mean = ' + str(serialYXMean2)
#    
#    logFileLocation = "../logs/halfwidth-3.log"
#    serialXYMean3 , serialYXMean3 = getSerialData(logFileLocation)
#    print 'For halfwidth 3 : Serial XY Mean = ' + str(serialXYMean3)  
#    print 'For halfwidth 3 : Serial YX Mean = ' + str(serialYXMean3)
    
#    logFileLocation = "../logs/halfwidth-2.log"
    logFileLocation = "/Users/Gaurav/Downloads/logmain.txt"
    
    parallelYXMeans, parallelXYMeans, parallelCoalMeans, mergedDualParallelMeans, dualParallelMeans = getParallelData(logFileLocation)
    
#    parallelYXMeans = collections.OrderedDict(sorted(parallelYXMeans.items()))
#    plt.plot(parallelYXMeans.keys(),[parallelYXMeans[1]/x for x in parallelYXMeans.values()],marker="o",label="smoothParallelYXFor")
#    plt.plot([1,32],[1,32],label="Linear (x=y)")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for smoothParallelYXFor')
#    plt.xlim(0,32)
#    plt.ylim(0,32)
#    plt.legend()
#    for key, value in parallelYXMeans.iteritems():
#        print key, parallelYXMeans[1]/value
    
#    parallelXYMeans = collections.OrderedDict(sorted(parallelXYMeans.items()))
#    plt.plot(parallelXYMeans.keys(),[parallelXYMeans[1]/x for x in parallelXYMeans.values()],marker="o",label="smoothParallelXYFor")
#    plt.plot([1,32],[1,32],label="Linear (x=y)")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for smoothParallelXYFor')
#    plt.xlim(0,32)
#    plt.ylim(0,32)
#    plt.legend()
#    for key, value in parallelXYMeans.iteritems():
#        print key, parallelXYMeans[1]/value
    
#    parallelCoalMeans = collections.OrderedDict(sorted(parallelCoalMeans.items()))
#    plt.plot(parallelCoalMeans.keys(),[parallelCoalMeans[1]/x for x in parallelCoalMeans.values()],marker="o",label="smoothParallelCoalescedFor")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for smoothParallelCoalescedFor')
#    plt.plot([1,32],[1,32],label="Linear (x=y)")
#    plt.xlim(0,32)
#    plt.ylim(0,32)
#    plt.legend()
#    for key, value in parallelCoalMeans.iteritems():
#        print key, parallelCoalMeans[1]/value

#    print parallelCoalMeans[1] - parallelXYMeans[1]
#    print parallelCoalMeans[2] - parallelXYMeans[2]
#    print parallelCoalMeans[4] - parallelXYMeans[4]
#    print parallelCoalMeans[8] - parallelXYMeans[8]
#    print parallelCoalMeans[16] - parallelXYMeans[16]
#    print parallelCoalMeans[32] - parallelXYMeans[32]    
#    

#    mergedDualParallelMeans = collections.OrderedDict(sorted(mergedDualParallelMeans.items()))
#    plt.plot(mergedDualParallelMeans.keys(),[mergedDualParallelMeans[1]/x for x in mergedDualParallelMeans.values()],marker="o",label="mergedDualSmoothParallelYXFor")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for mergedDualSmoothParallelYXFor')
#    plt.plot([1,32],[1,32],label="Linear (x=y)")
#    plt.xlim(0,32)
#    plt.ylim(0,32)
#    plt.legend()
#    for key, value in mergedDualParallelMeans.iteritems():
#        print key, mergedDualParallelMeans[1]/value
    
    dualParallelMeans = collections.OrderedDict(sorted(dualParallelMeans.items()))
    plt.plot(dualParallelMeans.keys(),[dualParallelMeans[1]/x for x in dualParallelMeans.values()],marker="o",label="dualSmoothParallelYXFor")
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (Ts/Tl)')
    plt.title('Speedup vs Number of Threads for dualSmoothParallelYXFor')
    plt.plot([1,32],[1,32],label="Linear (x=y)")
    plt.xlim(0,32)
    plt.ylim(0,32)
    plt.legend()
    for key, value in dualParallelMeans.iteritems():
        print key, dualParallelMeans[1]/value
    
    plt.show()
        
        