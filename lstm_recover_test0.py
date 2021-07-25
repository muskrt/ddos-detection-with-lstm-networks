#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import time
import random
import matplotlib.pyplot as plt
import Get_network_data 
from scipy.stats import itemfreq
import cv2
df=None
def preprocess_data(csv='tstveri.csv'):
    global df
    df=pd.read_csv(csv)
    df.loc[df['ttl']==254,'dport']=80
    df.loc[df['ttl']==254,'sport']=random.randint(0,63000)
    df.loc[df['ttl']==254,'src_ip']='.'
    df.to_csv("tstveri.csv",index=False)
    #for i in range(len(df.iloc[0,:])):
        #print(len(pd.unique(df.iloc[:,i])))
    df=df.drop(columns=['chksum','len','id'])
    #print("df values",df.iloc[0,:])
    #df = df[list(df.columns[~df.columns.duplicated()])]


def get_data(rank=None,input_flag=True,ft_indx=None):
    
    global df
    #df=pd.read_csv('test.csv')
    
    vector={}
    if input_flag==False :
        ft=[]
        with tf.device('/device:gpu:0'):
            if ft_indx !=0:
                ft=list(pd.unique(df.iloc[:,ft_indx]))
                ft.append('.')
            else:
                ft=list(pd.unique(df.iloc[:,ft_indx]))
        # with tf.device("/device:gpu:0"):
        #     for i in range(500):
        #         if i <len(ft):
        #             vector[i]=ft[i]
        #         else:
        #             vector[i]=-3

        # vector_array=[]

        char_to_idx = {c:i for (i,c) in enumerate(ft)}
        idx_to_char = {i:c for (i,c) in enumerate(ft)}
        return char_to_idx,idx_to_char
    elif input_flag:
        with tf.device('/device:gpu:0'):
            ft=list(pd.unique(df.iloc[:,rank].values))

        # with tf.device("/device:gpu:0"):
        #     for i in range(500):
        #         if i <len(ft):
        #             vector[i]=ft[i]
        #         else:
        #             vector[i]=-3

        # vector_array=[]

        char_to_idx = {c:i for (i,c) in enumerate(ft)}
        idx_to_char = {i:c for (i,c) in enumerate(ft)}
        vector_array=[]
        with tf.device("/device:gpu:0"):
            
            for i in range(df.shape[0]):
                ft_vector=[0 for i in range(500)]
                ft_vector[char_to_idx[df.iloc[i,rank]]]=1
                #print(ft_vector.index(1),ft[ft_vector.index(1)],str(df.iloc[i,rank]))
                vector_array.append(ft_vector)
        input_data=np.array(vector_array).reshape(150000,500)
        input_data=input_data.T
        # globals()['vector']=vector
        return input_data


class Lstm:
    def __init__(self,epochs=1000):
        cache=[]
        self.epochs=epochs
        self.losses=[]
        
        wf,wi,wg,wo=(np.random.rand(10,510) for i in range(4))
        wy=np.random.rand(500,10)
        params=(wf,wi,wf,wo,wy)
        
        globals()['params']=params
        globals()['cache']=cache
    def sigmoid(self,x):
        x=np.clip(x,1e-7,1e+7)
        return 1/(1+np.exp(-x))
    def tanh(self,x):
        return np.tanh(x)
        #return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def dsigmoid(self,x):
        return x*(1-x)
    def dtanh(self,x):
        return 1-x**2
    def clip(self,x):
        return np.clip(x,1e-7,1e+7)
    def forward(self,h_prev,c_prev,x,target=None):
        activations=[]
        h_prev=self.clip(h_prev)
        c_prev=self.clip(c_prev)
        x=self.clip(x)

        cache=globals()['cache']
        wf,wi,wg,wo,wy=globals()['params']
        
        z=np.row_stack((h_prev,x))

        
        f=self.sigmoid(np.dot(wf,z))
        i=self.sigmoid(np.dot(wi,z))
        g=self.tanh(np.dot(wg,z))
        o=self.sigmoid(np.dot(wo,z))
        
        c=np.multiply(f,c_prev)+np.multiply(i,g)
        h=np.multiply(o,self.tanh(c))
        
        v=np.dot(wy,h)


        y=np.exp(v)/np.sum(np.exp(v))

        y=self.clip(y)

        cell_cache=(z,f,i,g,o,h,c,y,target,h_prev,c_prev)
        cache.append(cell_cache)
        
        globals()['cache']=cache
        
        return h,c,y
        
    def backward(self):
        cache=globals()['cache']
        dh_next=np.zeros((10,1))
        #lr=1e-3
        lr=0.001
        dc_next=np.zeros((10,1))
        dwf,dwi,dwg,dwo,dwy=(0,0,0,0,0)
        wf,wi,wg,wo,wy=globals()['params']
        for t in reversed(range(len(cache))):
            z,f,i,g,o,h,c,y,target,h_prev,c_prev=cache[t]

            y_clip=np.clip(y,1e-7,1e+7)
            dv = np.copy(y_clip)
            dv[target] -= 1

            dwy += np.dot(dv, h.T)
  

            dh = np.dot(wy.T, dv)
            dh += dh_next
            dh=dh

            do = dh * self.tanh(c)
            do = self.dsigmoid(o) * do
            dwo += np.dot(do, z.T)


            dc = np.copy(dc_next)
            dc=np.clip(dc,1e-6,1e+6)
            # if nan in dc or nan in dh or nan in o or nan in self.dtanh(self.tanh(c)):
            #     print(dc,dh,o,self.dtanh(self.tanh(c)))
            #     sys.exit()
            if True in list(np.isnan(dc)):
                print("dc",dc)
            elif True in list(np.isnan(dh)):
                print("dh",dh)
            elif True in list(np.isnan(o)):
                print("0",o)
            elif True in list(np.isnan(self.dtanh(self.tanh(c)))):
                print("tanhc",self.tanh(c))
                print("dtanhc",self.dtanh(self.tanh(c)))
                sys.exit()

            dc += dh * o * self.dtanh(self.tanh(c))
            dg = dc * i
            dg = self.dtanh(g) * dg
            dwg += np.dot(dg, z.T)
       

            di = dc * g
            di = self.dsigmoid(i) * di
            dwi += np.dot(di, z.T)


            df = dc * c_prev
            df = self.dsigmoid(f) * df
            dwf += np.dot(df, z.T)
  

            dz = (np.dot(wf.T, df) 
                  + np.dot(wi.T, di) 
                  + np.dot(wg.T, dg) 
                  + np.dot(do.T, do))

            dh_next =dz[:10, :]
            dc_next = (f * dc)
        wf -= dwf*lr
        wf = np.clip(wf,1e-7,1e+7)
        
        wi -=dwi*lr
        wi =np.clip(wi,1e-7,1e+7)

        wg -=dwg*lr
        wg =np.clip(wg,1e-7,1e+7)
        
        wo -=dwo*lr
        wo =np.clip(wo,1e-7,1e+7)
        
        wy -=dwy*lr
        wy =np.clip(wy,1e-7,1e+7)
        params=(wf,wi,wg,wo,wy)
        globals()['params']=params
  
    def predict(self,grads):
        count=1
        while count:
            Get_network_data.Acquire_data()
            
            data=Get_network_data.dataframe
            
            data=pd.DataFrame(data)
            #print(data)
            with tf.device('/device:gpu:0'):
                data=self.encode(data)

            y=[[[] for j in range(11)] for i in range(20)]

            h=np.zeros((10,1))
            c=np.zeros((10,1))
            
            for i in range(20):
                #print(i)

                for j in range(11):
                    #print(j)
                    params=grads[j]
                    globals()['params']=params
                    output_values=[]
                    prediction=data[i,j]
                    for k in range(20):
                        #print(k)
                        h,c,output=self.forward(h,c,prediction.reshape(-1,1))
                        predicted=np.zeros((500,1))
                        predicted[np.argmax(output,axis=0)]=1
                        output_values.append(predicted)
                        prediction=predicted
                    y[i][j]=output_values
            #print(y)
            with tf.device('/device:gpu:0'):
                rt=self.filter(y)
                if rt:
                    count+=1
            count-=1

    def filter(self,Ddos_detect):
        features=[[[] for j in range(11) ] for i in range(20) ]
        
        for i in range(20):
            for j in range(11) :
                
                feature_1=self.decode(Ddos_detect[i][j],j)
                features[i][j]=feature_1

        ##find max counts
        max_feature_counts=[[] for i in range(11)]
        for i in range(20):
            for j in range(11):

                max_feature_counts[j].append(max(features[i][j],key=features[i][j].count))
        #print(max_feature_counts)
        dt=self.warning(max_feature_counts)
        if dt:
            return 1
    def warning(self,max_values):
        warning=0.0
        prob=[[] for i in range(len(max_values))]
        #print("girdi")
        for i in range(len(max_values)):
            prob[i].append(max(max_values[i],key=max_values[i].count))
        for i in prob:
            if '.' in i:
                warning+=2
            else:
                warning-=2
                
        if warning>15:
            print("----------------Ddos probability--------------> %",abs(warning*4))
            return 1
        else:
            print("-----------------Ddos probability-------------------> %",abs(warning*4))
            return 1

    def encode(self,data):
        data=np.array(data)
        
        with tf.device("/device:gpu:0"):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    char_to_idx,_=get_data(rank=None,input_flag=False,ft_indx=j)


                    vector=np.zeros((500,1))
                    if data[i,j] in char_to_idx:
                        vector[char_to_idx[data[i,j]]]=1
                    else:
                        data[i,j]='.'
                        vector[char_to_idx['.']]=1
                    data[i,j]=vector
        return data

    def decode(self,data,ft_id):
        ft=ft_id
 #       shape=(len(data),500)
        _,idx_to_char=get_data(rank=None,input_flag=False,ft_indx=ft)
#        data=np.array(data).reshape(shape)
        values=[]
        for i in data:
            key=list(i).index(1)
            if key in idx_to_char:
                value=idx_to_char[list(i).index(1)]
                values.append(value)
            else:
                value='.'
                values.append(value)
        return values

    def fit(self,train_data,rank):
        loss=0.0
        h=np.zeros((10,1))
        c=np.zeros((10,1))
        with tf.device("/device:gpu:0"):
            
            for j in range(500):
                targets=[]
                y=[]
                loss=0.0
                #(j%500)*100+1,((j%500)+1)*100-1
                # if j==48 and count==0:
                #     count+=1
                #     j=0 
                # elif count==1 and j==48:
                #     self.plot_loss()
                #     break
                for i in range(149998):
                    
                    target=list(train_data[:,i+1]).index(1)
                    h,c,output=self.forward(h,c,train_data[:,i].reshape(-1,1),target)
                    y.append(output)
                    targets.append(list(train_data[:,i+1]))
                    loss+= -np.log(np.clip(output[target],1e-7,1e+7))

                    if i %20==0:
                        
                        self.backward()
                        cache=[]
                        #loss=0.0
                        globals()['cache']=cache
                
                self.losses.append(loss/150000)

                self.get_acc(y,targets,loss/150000,j,rank)
                #self.losses.append(loss/50000)
        self.plot_loss()
        self.save_weights(rank)
        # for epoch in range(self.epochs):
        #     cache=[]
        #     globals()['cache']=cache
            
        #     h,c,y=self.forward(h0,c0,inputs)
        #     loss+= -np.log(y[target])
        #     self.backward()
            
    def plot_loss(self):
        plt.plot(self.losses)
        plt.show()
    def save_weights(self,rank):
        params=globals()['params']
        wf,wi,wg,wo,wy=params
        np.savez("feuature"+str(rank)+".npz",Wf=wf,Wi=wi,Wg=wg,Wo=wo,Wy=wy)
    def get_acc(self,y,targets,loss,j,rank):
        acc=[0 for i in range(len(y))]
        for i in range(len(y)):
            if np.argmax(y[i],axis=0)==targets[i].index(1):
                acc[i]=1
        print("rank",rank,"Epoch:",j,"Loss",loss,"Acc",np.mean(acc))
def Grads():
    grad0=np.load("feuature0.npz")
    grad1=np.load("feuature1.npz")
    grad2=np.load("feuature2.npz")
    grad3=np.load("feuature3.npz")
    grad4=np.load("feuature4.npz")
    grad5=np.load("feuature5.npz")
    grad6=np.load("feuature6.npz")
    grad7=np.load("feuature7.npz")
    grad8=np.load("feature8.npz")
    grad9=np.load("feuature9.npz")
    grad10=np.load("feuature10.npz")
    grad0=(grad0['Wf'],grad0['Wi'],grad0['Wg'],grad0['Wo'],grad0['Wy'])
    grad1=(grad1['Wf'],grad1['Wi'],grad1['Wg'],grad1['Wo'],grad1['Wy'])
    grad2=(grad2['Wf'],grad2['Wi'],grad2['Wg'],grad2['Wo'],grad2['Wy'])
    grad3=(grad3['Wf'],grad3['Wi'],grad3['Wg'],grad3['Wo'],grad3['Wy'])
    grad4=(grad4['Wf'],grad4['Wi'],grad4['Wg'],grad4['Wo'],grad4['Wy'])
    grad5=(grad5['Wf'],grad5['Wi'],grad5['Wg'],grad5['Wo'],grad5['Wy'])
    grad6=(grad6['Wf'],grad6['Wi'],grad6['Wg'],grad6['Wo'],grad6['Wy'])
    grad7=(grad7['Wf'],grad7['Wi'],grad7['Wg'],grad7['Wo'],grad7['Wy'])
    grad8=(grad8['Wf'],grad8['Wi'],grad8['Wg'],grad8['Wo'],grad8['Wy'])
    grad9=(grad9['Wf'],grad9['Wi'],grad9['Wg'],grad9['Wo'],grad9['Wy'])
    grad10=(grad10['Wf'],grad10['Wi'],grad10['Wg'],grad10['Wo'],grad10['Wy'])
    return [grad0,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10]

    return grads
if __name__=="__main__":
    preprocess_data(csv='normal_veri.csv')
    model=Lstm()
    

    rank=0
    feature_id=rank-1
    if rank==0:
        grads=Grads()
        #input_data=get_data(0)
        preprocess_data()
        #cv2.waitKey(60000*60*60*2)
        with tf.device("/device:gpu:0"):
            model.predict(grads)
    if rank !=0:
        #cv2.waitKey(1000)
        input_data=get_data(feature_id,input_flag=True)
        with tf.device("/device:gpu:0"):
            model.fit(input_data,rank)
        #model.predict(input_data[:,0])
    