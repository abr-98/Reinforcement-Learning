import random
import numpy as np
from collections import namedtuple,deque

class replayBuffer:
    transition=namedtuple('Transition',['s','a','r','s_','nd'])
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=deque([],maxlen=self.capacity)
        
    def push(self,s,a,r,s_,nd):
        tr=replayBuffer.transition(np.float32(s),a,r,np.float32(s_),nd)
        self.memory.append(tr)
        
    def sample(self,batch_size):
        tr_batch=random.choices(self.memory,k=batch_size)
        s=[];a=[];r=[];s_=[];nd=[]
        for tr in tr_batch:
            s.append(tr.s);a.append(tr.a);r.append(tr.r),s_.append(tr.s_),nd.append(tr.nd)
        return np.array(s),np.array(a),np.array(r),np.array(s_),np.uint8(nd)