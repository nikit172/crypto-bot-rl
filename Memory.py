import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers
import pprint
import logging
from logging import info,warning,error,debug
import PPO_model2
from keras.src import ops




@keras.saving.register_keras_serializable('Memory')
class Memory(keras.layers.Layer):
    def __init__(self,N,D_k,D_v,name_='15',**kwargs):
        super().__init__(**kwargs)
        """args:
        
        N: num slot in memory
        
        D: dim in slot"""
        self.N=N
        self.D=D_k
        self.D_v=D_v
        self.name_=name_
        
        # def _w(shape,init,ind):
        #     nam= val_names[ind] if val_names and ind< len(val_names) else None
        #     return self.add_weight(shape,init,tf.float32,trainable=False,name=nam)
        

        # self.memory_keys=_w((N,D_k),keras.initializers.random_normal(0,0.1),0)
        # self.memory_vals=_w((N,D_v),keras.initializers.random_normal(0,0.1),1)
        # self.memory_importent=_w((N,),keras.initializers.random_normal(0,0.1),2)
        # self.memory_age=_w((N,),keras.initializers.zeros(),3)
        # self.memory_usage=_w((N,),keras.initializers.zeros(),4)
        # self.adding=tf.ones((self.N,),tf.float32)
        self.memory_keys=self.add_weight((N,D_k),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False,)
        self.memory_vals=self.add_weight((N,D_v),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False,)
        self.memory_importent=self.add_weight((N,),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False)
        self.memory_age=self.add_weight((N,),keras.initializers.zeros(),tf.float32,trainable=False,)
        self.memory_usage=self.add_weight((N,),keras.initializers.zeros(),tf.float32,trainable=False)
        self.adding=tf.ones((self.N,),tf.float32)
        # """self.memory_keys=self.add_weight((N,D_k),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False,name=f'memory_keys_{name_}')
        # self.memory_vals=self.add_weight((N,D_v),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False,name=f'memory_vals_{name_}')
        # self.memory_importent=self.add_weight((N,),keras.initializers.random_normal(0,0.1),tf.float32,trainable=False,name=f'memory_imp_{name_}')
        # self.memory_age=self.add_weight((N,),keras.initializers.zeros(),tf.float32,trainable=False,name=f'memory_age_{name_}')
        # self.memory_usage=self.add_weight((N,),keras.initializers.zeros(),tf.float32,trainable=False,name=f'memory_usg_{name_}')"""
        self.built=True
        
    @property
    def get_keys(self):
        return tf.identity(self.memory_keys)
    @property
    def get_vals(self):
        return tf.identity(self.memory_vals)
    @property
    def get_impotent(self):
        return tf.identity(self.memory_importent)
    @property
    def get_age(self):
        return tf.identity(self.memory_age)
    @property
    def get_usage(self):
        return tf.identity(self.memory_usage)
    
    def read_slot(self,slot):
        "return mem_key, mem_val, mem_impo, mem_age, mem_usage"
        return (self.memory_keys[slot],self.memory_vals[slot],
                self.memory_importent[slot],
                self.memory_age[slot],self.memory_usage[slot])
    
    def write_slot(self,ind,key,val,importent,n):
        
        ind=tf.tile(tf.reshape(tf.cast(ind,tf.int32),(-1,1)),(n,1))
        key=tf.reshape(key,(n,self.D))
        val=tf.reshape(val,(n,self.D_v))
        importent = tf.reshape(importent, (n,))
    
        
        memory_keys=tf.tensor_scatter_nd_update(self.memory_keys,ind,key)
        memory_vals=tf.tensor_scatter_nd_update(self.memory_vals,ind,val)
        memory_importent=tf.tensor_scatter_nd_update(self.memory_importent,ind, tf.reshape(importent,[-1]))
        memory_age=tf.tensor_scatter_nd_update(self.memory_age,ind,tf.zeros((n,), dtype=self.memory_age.dtype))
        memory_usage=tf.tensor_scatter_nd_update(self.memory_usage,ind,tf.zeros((n,), dtype=self.memory_age.dtype))
        return (memory_keys,
                memory_vals,
                memory_age,
                memory_usage,
                memory_importent)
    
    def update_slot_ema(self,ind,new_key,new_val,new_imp,lr,alpha_im,n):
        
        ind=tf.reshape(tf.cast(ind,tf.int32),(-1,1))

        old_key=tf.gather(self.memory_keys,ind[:,0])
        old_val=tf.gather(self.memory_vals,ind[:,0])
        old_imp=tf.gather(self.memory_importent,ind[:,0])


        memory_keys=tf.tensor_scatter_nd_update(self.memory_keys,ind,(1-lr)*old_key+lr*new_key)
        memory_vals=tf.tensor_scatter_nd_update(self.memory_vals,ind,(1-lr)*old_val+lr*new_val)
        
        memory_importent=tf.tensor_scatter_nd_update(self.memory_importent,ind,(1-alpha_im)*old_imp+alpha_im*tf.reshape(new_imp,[-1]))
        memory_age=tf.tensor_scatter_nd_update(self.memory_age,ind,tf.zeros((n,), dtype=self.memory_age.dtype))
        memory_usage=tf.tensor_scatter_nd_add(self.memory_usage, ind, tf.ones((n,), dtype=self.memory_usage.dtype))
        return (memory_keys,
                memory_vals,
                memory_age,
                memory_usage,
                memory_importent)

    def increment_age(self):
        self.memory_age.assign_add(self.adding)
    
    def increment_usage(self,inds):
        self.memory_usage.assign(tf.tensor_scatter_nd_add(self.memory_usage,tf.expand_dims(inds,-1),tf.ones(tf.shape(inds),tf.float32)))
        
    def get_normal_keys(self):
        return tf.nn.l2_normalize(self.memory_keys,-1)
    
    def get_volue(self,ind):
        return tf.gather(self.memory_vals,ind)
    
    def get_eviction_scores(self,alpha=1,beta=1,gamma=1,eps=1e-6):
        max_age =tf.reduce_max(self.memory_age)
        max_usage =tf.reduce_max(self.memory_usage)
        max_impot =tf.reduce_max(self.memory_importent)

        return alpha*self.memory_age/(max_age+eps) - beta*self.memory_usage/(max_usage+eps) - gamma*self.memory_importent/(max_impot+eps)
    
    
    def get_state(self):
        return {'keys':tf.identity(self.memory_keys),
                'val':tf.identity(self.memory_vals),
                'age':tf.identity(self.memory_age),
                'usage':tf.identity(self.memory_usage),
                'importent':tf.identity(self.memory_importent)}
    
    def set_state(self,state):
        self.memory_keys.assign(state['keys'])
        self.memory_vals.assign(state['val'])
        self.memory_age.assign(state['age'])
        self.memory_usage.assign(state['usage'])
        self.memory_importent.assign(state['importent'])

    def set_state_tuple(self,state):
        self.memory_keys.assign(state[0])
        self.memory_vals.assign(state[1])
        self.memory_age.assign(state[2])
        self.memory_usage.assign(state[3])
        self.memory_importent.assign(state[4])

    

    def get_config(self):
        conf=super().get_config()  
        base={'N':self.N,
              'D':self.D,
              'D_v':self.D_v,
              'Name':self.name_,}
        return {**base,**conf}
    
    @classmethod
    def from_config(cls,conf):
        N=conf.pop('N')
        D=conf.pop('D')
        D_v=conf.pop('D_v')
        Nam=conf.pop('Name')
        return cls(N,D,D_v,name_=Nam,**conf)

"""@classmethod
    def from_config(cls,conf):
        N=conf.pop('N')
        D=conf.pop('D')
        D_v=conf.pop('D_v')
        Nam=conf.pop('Name')
        state={'keys':conf.pop('keys'),'val':conf.pop('val'),'age':conf.pop('age'),'usage':conf.pop('usage'),'importent':conf.pop('importent')}
        cl=cls(N,D,D_v,name_=Nam,**conf)
        cl.set_state(state)
        return cl"""

@keras.saving.register_keras_serializable('Memory')
class ReadMemory(keras.layers.Layer):
    def __init__(self,memory: Memory,D_v,k=8,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.memory=memory
        self.D_v=D_v
        self.k=k

    def build(self, input_shape):
        self.D=input_shape[-1]
        if self.D!=self.D_v:
            raise ValueError(f'shape d and d_v not correct (need D==D_v) yuo:{input_shape}')
        return super().build(input_shape)
    
    def call(self,x,can_r,vector_h):
        can_r = tf.reshape(tf.reduce_any(can_r), []) 
        m=tf.cond(can_r,lambda : self.colculate_h(x), lambda : vector_h)
        return m
    
    def colculate_h(self,x):
        x_norm=tf.nn.l2_normalize(x,-1)
        keys=self.memory.get_normal_keys()
        s=tf.matmul(x_norm,keys,transpose_b=True)

        top_s,top_indexs=tf.math.top_k(s,self.k)

        self.memory.increment_usage(top_indexs)
        v_j=self.memory.get_volue(top_indexs)
        top_s_norm=tf.nn.softmax(top_s*10+1e-5)

        m=tf.reduce_sum(tf.expand_dims(top_s_norm,-1)*v_j,-2)
        return m
    # def compute_output_spec(self, x, can_r=None, vector_h=None):
    #     # Выход – тензор формы (batch_size, read_dim)
    #     return tf.keras.KerasTensor(shape=(None, self.D_v), dtype=x.dtype)
    
    def get_config(self):
        conf=super().get_config()
        base={'mem':keras.saving.serialize_keras_object(self.memory),
              'D_v':self.D_v,
              'k':self.k}
        return {**base,**conf}
    
    @classmethod
    def from_config(cls,conf):
        memory=keras.saving.deserialize_keras_object(conf.pop('mem'))
        d_v=conf.pop('D_v')
        k=conf.pop('k')
        return cls(memory,d_v,k,**conf)

        

        

#class write
@keras.saving.register_keras_serializable('Memory')
class WriteMemory(keras.layers.Layer):
    def __init__(self,memory: Memory,tf_name='15m',o_f=0.6,n_f=0.02,a_f=1,s_max=0.65,a=1,b=1,g=1,**kwargs):
        ''' o_f Пороги важности:15m: 0.4

        4h: 0.6

        1w: 0.8
        
        n_f Коэффициенты обучения для EM: 15m: 0.05

        4h: 0.02

        1w: 0.005

        a_f Коэффициент сглаживания важности: как правильо 1 для всех tf
        
        s_max Порог сходства для обновления = 0.6…0.7 (можно начать с 0.65 и подбирать).
        
        a,b,g Веса для eviction= для чоединения 3 значений из памяти'''
        super().__init__(**kwargs)
        self.memory=memory
        self.tf_name=tf_name
        self.o_f=o_f
        self.n_f=n_f
        self.a_f=a_f
        self.s_max=s_max
        self.a=a
        self.b=b
        self.g=g
    #build model
    def build(self, input_shape):
        self.D_k=self.memory.D
        self.D_v=self.memory.D_v

        self.KeyNet=layers.Dense(self.D_k,kernel_regularizer=keras.regularizers.l2(1e-4))
        self.ValueNet=layers.Dense(self.D_v,kernel_regularizer=keras.regularizers.l2(1e-4))
        self.ImpotentNet=layers.Dense(1,'sigmoid',kernel_regularizer=keras.regularizers.l2(1e-4))
        
        return super().build(input_shape)
    
    def check_batch(self,x):
        # Действие, если batch_size != 1 и не training
        tf.debugging.assert_equal(tf.shape(x)[0], 1, 
                                   message="dont trainable==True and batch_size!=1")
        return x  # или любое другое действие

    def no_check(self,x):
        return x

    def call(self,x,training=None,can_write=None):
        training = training if training is not None else False
        can_write = can_write if can_write is not None else False
        can_write = tf.reshape(tf.reduce_any(can_write), [])
        # x=tf.cond(
        # tf.logical_and(tf.not_equal(tf.shape(x)[0], 1), tf.logical_not(training)),
        # lambda:self.check_batch(x),
        # lambda:self.no_check(x)
        # )
        k_new=tf.nn.l2_normalize(self.KeyNet(tf.stop_gradient(x)),-1)
        v_new=self.ValueNet(tf.stop_gradient(x))
        p=self.ImpotentNet(tf.stop_gradient(x))
        #tf.print(tf.shape(x)[0],'| can_wite:',can_write,'| training:',tf.logical_not(tf.cast(training,tf.bool)),'| other:',tf.equal(tf.shape(x)[0], 1),'| ',tf.cast(tf.squeeze(p>self.o_f),tf.bool))
        
        pr=tf.logical_and(tf.equal(tf.shape(x)[0], 1),can_write)
        return tf.cond(pr,lambda: self.batch_1(training,x,p,k_new,v_new,can_write),lambda:(k_new,v_new,p))
    
    def batch_1(self,training,x,p,k_new,v_new,can_write):
        # k_new = k_new[0:1]               # (1, dim)
        # v_new = v_new[0]                  # (dim,)
        # p = p[0]
        condition=tf.logical_and(
            tf.logical_and(tf.logical_not(tf.cast(training,tf.bool)), 
                          tf.logical_and( tf.equal(tf.shape(x)[0], 1),
                                          tf.cast(tf.reshape(p>self.o_f,(-1,)),tf.bool))),
            can_write)
        
        return tf.cond(condition,lambda :self.update_memory(k_new,v_new,p),lambda:(k_new,v_new,p))
    
    # def compute_output_spec(self, x, can_write=None):
    #     key_shape = self.KeyNet.compute_output_spec(x).shape
    #     value_shape = self.ValueNet.compute_output_spec(x).shape
    #     imp_shape = self.ImpotentNet.compute_output_spec(x).shape
    #     #return (tf.TensorSpec(key_shape,tf.float32),tf.TensorSpec(value_shape,tf.float32),tf.TensorSpec(imp_shape,tf.float32))
    #     return (
    #         tf.keras.KerasTensor(shape=key_shape, dtype=x.dtype),
    #         tf.keras.KerasTensor(shape=value_shape, dtype=x.dtype),
    #         tf.keras.KerasTensor(shape=imp_shape, dtype=x.dtype),
    #         )
    def update_memory(self,k_new,v_new,p):
        n=tf.shape(p)[0]
        similarity=tf.matmul(k_new,self.memory.get_normal_keys(),transpose_b=True)
        
        #top_v,top_i=tf.reduce_max(similarity,-1),tf.argmax(similarity,-1)
        top_v, top_i = tf.math.top_k(similarity, k=1)

        slot=tf.reduce_any(top_v>self.s_max)
        
        mem_state=tf.cond(slot,lambda : self.update_slot(top_i,k_new,v_new,p,n),lambda: self.replase_slot(k_new,v_new,p,n))
        self.memory.set_state_tuple(mem_state)
    
        return (k_new,v_new,p)


    # slot in memory replace
    def replase_slot(self,k_new,v_new,p,n):
        memory_stronger=self.memory.get_eviction_scores(self.a,self.b,self.g)
        # ind=tf.argmax(memory_stronger)
        _,ind=tf.math.top_k(memory_stronger,k=1)
        mem=self.memory.write_slot(ind,k_new,v_new,p,n)

        return mem
        

    def update_slot(self,ind,new_key,new_val,new_imp,n):
         return self.memory.update_slot_ema(ind,new_key,new_val,new_imp,self.n_f,self.a_f,n)
    
    def get_config(self):
        conf=super().get_config()
        #memory,tf_name='15m',o_f=0.6,n_f=0.02,a_f=1,s_max=0.65,a=1,b=1,g=1,
        base={'mem':keras.saving.serialize_keras_object(self.memory),
              'tf_name':self.tf_name,
              'o_f':self.o_f,
              'n_f':self.n_f,
              'a_f':self.a_f,
              's_max':self.s_max,
              'a':self.a,
              'b':self.b,
              'g':self.g}
        return{**base,**conf}
    
    @classmethod
    def from_config(cls,conf):
        mem=keras.saving.deserialize_keras_object(conf.pop('mem'))
        tf_name=conf.pop('tf_name')
        o_f=conf.pop('o_f')
        n_f=conf.pop('n_f')
        a_f=conf.pop('a_f')
        s_max=conf.pop('s_max')
        a=conf.pop('a')
        b=conf.pop('b')
        g=conf.pop('g')
        return cls(mem,tf_name,o_f,n_f,a_f,s_max,a,b,g,**conf)
        
        



    
    

        