def getmentiondistance(mentionbeginindex,mentionendindex,candbeginindex, candendindex):
    if mentionbeginindex== candbeginindex:
        return 0
    if mentionbeginindex>candendindex:
        return mentionbeginindex-candendindex
    else:
        return candbeginindex-mentionendindex
def addToContextMap(distance,context,cand):
    free = False
    while not free:
        if not distance in context:
            context[distance] = cand
            free= True
        else:
            distance=distance+1
    return context

def getmentioncontext(mention,contextmentions,maxcontextlen):
    mentioncontext={}
    for cand in contextmentions:
        if cand.taIdentRef is not None:
            distance=getmentiondistance(int(mention.begin_index),int(mention.end_index),int(cand.begin_index),int(cand.end_index))
            if distance>0:
                if len(mentioncontext)<maxcontextlen:
                    mentioncontext=addToContextMap(distance,mentioncontext,cand)
                else:
                    max = 0
                    for k in mentioncontext.keys():
                        if distance<k and max <k:
                            max=k
                    if max>0:
                        mentioncontext.pop(max)
                        mentioncontext=addToContextMap(distance,mentioncontext,cand)
    context=[]
    for id in sorted(mentioncontext):
        context.append(mentioncontext[id].uri)
    return context