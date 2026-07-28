import numpy as np
from tov import *
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool,Process, Queue
from matplotlib import pyplot as plt
import time
# 多线程tov
class multithreadtov:
    def __init__(self,eos,poolnum=32,stepsize=0.1,dr=0.001):
        self.eos=eos
        self.stepsize=stepsize
        self.tovfinalresult=[]
        self.dr=dr
        self.poolnum=poolnum
        self.datanum=len(eos)
        if self.datanum<poolnum:
            self.poolnum=self.datanum
        self.progressnum=np.zeros(self.datanum)
        # self.tovsinglelength=int(np.max(eos[0,:,0])/stepsize)+1
        self.tovsinglelength=200

    def callback(self,result): 
        # print('callback')
        try:
            num=result[0]
            tovdata=result[1]
            tovdata=tovdata
            # print(tovdata.shape)
            # print(num)
            self.tovfinalresult[num,:,0:len(tovdata[0,0])]=tovdata
            self.progressnum[num]=1
            # print("\r{:3}%".format(sum(self.progressnum)/self.datanum),end=' ')

        except Exception as e:
            print('callback',e)

    def tovonce(self,nummm,queue):
        try:
           
            tovdata=np.zeros((len(nummm),2,self.tovsinglelength))
            
            for num in nummm:
                stepsize=self.stepsize
                eossingle=self.eos[num]
                eossingle=eossingle[eossingle[:,2]>0]
                eossingle=eossingle[eossingle[:,1]>0]
                rhomax=np.max(eossingle[:,0])
                tovdata2=main(eossingle,(rhomax-0.8*0.16)/(0.16*stepsize),stepsize*0.16,dr=self.dr)
                tovdata[num-nummm[0],:,0:len(tovdata2[0])]=np.array([tovdata2[0],tovdata2[1]])
                queue.put(1)
            #print(tovdata)
        except Exception as e:
            print('tovonce',e)
        return nummm,tovdata
    def progress(self,queue):#进度条
        time.sleep(5)#避免进度条冲突
        bar= tqdm(total=self.datanum,desc='calulating tov')
        while bar.n<self.datanum:
            a=0
            try:
                a=queue.get(block=False)
            except Exception as e:
                pass
            bar.update(a)
            time.sleep(0.1)

#开始计算
    def startcalu(self):
        
        queue = mp.Manager().Queue()
        # pbar=tqdm(total=self.datanum)
        # callback = lambda *args:(self.callback,pbar.update)
        MM=self.datanum
        poolnum=self.poolnum
        # data,eos=geneos(MM,net1,net2)
        self.tovfinalresult=np.zeros((MM,2,self.tovsinglelength))
        pool = Pool(poolnum+1)
        slicepool=int(MM/poolnum)
        
        for i in tqdm(range(poolnum),desc='distributing workload to '+str(self.poolnum)+' threads'):
            if (i==poolnum-1):
                end=MM
            else:
                end=(i+1)*slicepool
            # print('start',i*slicepool,'end',end)
            pool.apply_async(self.tovonce, callback=self.callback,args=(range(i*slicepool,end),queue,))
        pool.apply_async(self.progress,args=(queue,))  
        # self.bar=tqdm(total=self.datanum,desc='calulating tov')
        pool.close()
        pool.join()
        return self.tovfinalresult