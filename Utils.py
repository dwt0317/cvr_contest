

def list2dict(mylist):
    myDict = {}
    i = 0
    for v in mylist:
        myDict[v] = i
        i += 1
    return myDict