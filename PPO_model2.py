import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers
import pprint
import logging
from logging import info,warning,error,debug
from keras.src import ops


@keras.saving.register_keras_serializable('RMSNorm')
class RMSNorm(layers.Layer):
    def __init__(self,eps=1e-5,**kwargs):
        """вычесляет RMSNorm(x/sqrt(mean(x**2)) для более простой
        и быстрой нармолизации без потерь в качестве"""

        super().__init__(**kwargs)
        self.eps=eps
    def build(self, input_shape):
        self.dim=input_shape[-1]
        self.weight = self.add_weight(
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
            name="weight_RMS"
        )
        return super().build(input_shape)
    
    def call(self,x):
        #tf.print(f'rmsnorm:{tf.reduce_any(tf.math.is_nan(x))}|mean:{tf.reduce_mean(x)},max:{tf.reduce_min(x)}, min:{tf.reduce_max(x)}',end='|')
        rms=tf.reduce_mean(tf.square(x),axis=-1,keepdims=1)
        rms=tf.sqrt(tf.maximum(rms+self.eps,1e-5))
        y=x/rms
        #tf.print(f"{tf.reduce_any(tf.math.is_nan(y))}|mean:{tf.reduce_mean(y)},max:{tf.reduce_min(y)}, min:{tf.reduce_max(y)}")
        return y*self.weight
    
    def get_config(self):
        base_conf = super().get_config()
        config={
            'eps':self.eps,
        }
        return {**base_conf,**config}
    
    @classmethod
    def from_config(cls,config):
        eps=config.pop('eps')
        return cls(eps,**config)
    def compute_output_shape(self, input_shape):
        # output shape = input shape (batch, seq, d_model)
        return tf.TensorShape(input_shape)



class RelativatePos(layers.Layer):
    def __init__(self,head=1):
        """позиционирование для трансформера"""
        super().__init__()
        self.head=head
    
    def build(self, input_shape):
        self.shape=input_shape
        if self.shape[-3]!=self.head:
            raise ValueError(f"неверный shape; указан head:{self.head}, а введены данные с head:{self.shape[-3]}")
        self.T=self.shape[-2]
        self.L=2*self.T-1
        self.weight=self.add_weight((self.head,self.L),dtype=tf.float32)

    def call(self,x:tf.TensorArray) -> tf.TensorArray:
        p=tf.range(0,self.T,1,tf.int32)
        delta=(p[None,:]-p[:,None])
        ind=delta+(self.T-1)
        pos=tf.gather(self.weight,ind,axis=1,batch_dims=0)
        return pos
    
    
@keras.saving.register_keras_serializable(package='crypto_bot')
class ffa(layers.Layer):
    def __init__(self,d_ff,num_expert,**kwargs):
        super().__init__(**kwargs)
        self.d_ff=d_ff
        self.num_expert=num_expert

    def build(self,shape):
        self.d_model=shape[-1]
        self.w1=self.add_weight((self.num_expert,self.d_model,self.d_ff),
                                initializer=keras.initializers.glorot_uniform,
                                trainable=True,
                                regularizer=keras.regularizers.l2(1e-5))
        self.b1=self.add_weight((self.num_expert,self.d_ff),
                                initializer=keras.initializers.glorot_uniform,
                                trainable=True,
                                regularizer=keras.regularizers.l2(1e-5))
        self.w2=self.add_weight((self.num_expert,self.d_ff,self.d_model),
                                initializer=keras.initializers.glorot_uniform,
                                trainable=True,
                                regularizer=keras.regularizers.l2(1e-5))
        self.b2=self.add_weight((self.num_expert,self.d_model),
                                initializer=keras.initializers.glorot_uniform,
                                trainable=True,
                                regularizer=keras.regularizers.l2(1e-5))
        self.dropount=keras.layers.Dropout(0.15)
        super().build(shape)
        
    def call(self,x,expert_id,expert_val,training=None):
        w1=tf.gather(self.w1,expert_id)
        b1=tf.gather(self.b1,expert_id)
        w2=tf.gather(self.w2,expert_id)
        b2=tf.gather(self.b2,expert_id)
        ##tf.print('shape x:',tf.shape(x),'| w1 shape:',tf.shape(w1),'| b1 shape:',tf.shape(b1),'| expert_val:',tf.shape(expert_val))
        y=tf.squeeze(tf.matmul(tf.expand_dims(x,1),w1))+b1
        y=self.dropount(keras.activations.gelu(y),training=training)
        y=tf.squeeze(tf.matmul(tf.expand_dims(y,1),w2))+b2
        y*=tf.expand_dims(expert_val,-1)
        return y
    
    def compute_output_shape(self, input_shape):
        # output shape = input shape (batch, seq, d_model)
        return tf.TensorShape(input_shape)

    
    def get_config(self):
        base_conf = super().get_config()
        config={
            'd_ff':self.d_ff,
            'num_expert':self.num_expert
        }
        return {**base_conf,**config}
    
    @classmethod
    def from_config(cls,config):
        d_ff=config.pop('d_ff')
        num_expert=config.pop('num_expert')
        return cls(d_ff,num_expert,**config)


@keras.saving.register_keras_serializable(package='crypto_bot')
class Moe(layers.Layer):
    ''' need add:
        load balancing loss

        importance loss

        entropy penalty'''
    def __init__(self,d_ff,num=0,num_expert=8,top_k=2,l=3,**kwargs):
        ''' d_model: output dim from ffa 
            d_ff: inside dim in ffa
            num: num transformer for regulator
            num_expert: num expert in moe
            top_k: num expert ffa to one step
            l: num all transformer'''
        super().__init__()
        self.num_expert=num_expert
        self.num=num
        self.top_k=top_k
        self.d_ff=d_ff
        self.l=l
        self.metric1=keras.metrics.Mean(name=f'rou_ent_{self.num}')
        self.metric2=keras.metrics.Mean(name=f'impotent_{self.num}')
    
#     def build(self, input_shape):
#         super().build(input_shape)
#         self.shape=input_shape
#         #self.expert=[ffa(self.d_ff) for _ in range(self.num_expert)]
#         self.ffa=ffa(self.d_ff,num_expert=self.num_expert)
#         self.ffa.build(input_shape)
        
#         self.router= layers.Dense(self.num_expert,kernel_regularizer=keras.regularizers.l2(1e-5),name='router')
#         self.router.build(input_shape) 
#         self.expert_ids = self.add_weight(
#         name="expert_ids",
#         shape=(self.num_expert,),
#         dtype=tf.int32,
#         initializer=tf.keras.initializers.Constant(
#             list(range(self.num_expert))
#         ),
#         trainable=False,
# )
    def build(self, input_shape):
        super().build(input_shape)
        self.shape=input_shape
        self.expert=ffa(self.d_ff,self.num_expert)
        self.expert.build(input_shape)
        
        self.router= layers.Dense(self.num_expert,kernel_regularizer=keras.regularizers.l2(1e-5),name='router')
        self.router.build(input_shape) 
        self.expert_ids = self.add_weight(
        name="expert_ids",
        shape=(self.num_expert,),
        dtype=tf.int32,
        initializer=tf.keras.initializers.Constant(
            list(range(self.num_expert))
        ),
        trainable=False,
        )
        self.dropont=keras.layers.Dropout(0.15)
    
    def call(self,x,training=None,step=None):
        #tf.print(f'moe:{tf.reduce_any(tf.math.is_nan(x))}|mean:{tf.reduce_mean(x)},max:{tf.reduce_min(x)}, min:{tf.reduce_max(x)}')
        num_tokens=tf.shape(x)[0]*tf.shape(x)[1]
    
        logit=self.router(x,training=training)
        #tf.print(f'logit:{tf.reduce_any(tf.math.is_nan(logit))}|mean:{tf.reduce_mean(logit)},max:{tf.reduce_min(logit)}, min:{tf.reduce_max(logit)}',end='|')

        logit+=tf.cond(tf.logical_and( training is not False, step is not False),lambda : self.nose(logit),lambda: tf.zeros_like(logit))
        gates=tf.nn.softmax(logit,-1)
        #tf.print(f'gates:{tf.reduce_any(tf.math.is_nan(gates))}|mean:{tf.reduce_mean(gates)},max:{tf.reduce_min(gates)}, min:{tf.reduce_max(gates)}',end='|')
        
        top_k_value,self.top_k_index=tf.math.top_k(gates,self.top_k,index_type=tf.int32)
        top_k_value=tf.nn.softmax(top_k_value,-1)
        ##tf.print(f'top_k shape: {tf.shape(top_k_value)} top_k:',tf.shape(top_k_value))#-----------------------------------------------------
        #-------------new call---------------------------------------
        
        x_pred = tf.reshape(x, (num_tokens, self.shape[-1]))
        ##tf.print('expert_id shape:',tf.shape(self.top_k_index),' | ',(num_tokens*self.top_k))
        ##tf.print('expert_prob shape:',tf.shape(top_k_value),'| ',(num_tokens*self.top_k))

        expert_id = tf.reshape(self.top_k_index, (-1,))
        expert_prob = tf.reshape(top_k_value, (-1,))
        x_pred = tf.repeat(x_pred, repeats=self.top_k, axis=0)

        output= self._ffa(x_pred,expert_id,expert_prob,training=training)#self._ffa_predict() # добавить параметры
        #tf.print(f'out:{tf.reduce_any(tf.math.is_nan(output))}|mean:{tf.reduce_mean(output)},max:{tf.reduce_min(output)}, min:{tf.reduce_max(output)}',end='|')
        
        ##tf.print('shape output:',tf.shape(output))
        y_pred=tf.reshape(output,(num_tokens,self.top_k,self.shape[-1]))

        y_pred=tf.reduce_sum(y_pred,axis=-2)
        y_pred=tf.reshape(y_pred,(-1,self.shape[1],self.shape[2]))

        #------------my_loss--------------
        losses,loss1,loss2=self._loss(gates,num_tokens)
        if training is not False:
            self.add_loss(losses)
            self.metric1.update_state(loss1)    
            self.metric2.update_state(loss2)    
        #tf.print(f'y_pred:{tf.reduce_any(tf.math.is_nan(y_pred))}|mean:{tf.reduce_mean(y_pred)},max:{tf.reduce_min(y_pred)}, min:{tf.reduce_max(y_pred)}')
        
        return y_pred
    
    def nose(self,logit):
        variance = tf.reduce_mean(tf.square(tf.stop_gradient(logit) - tf.reduce_mean(tf.stop_gradient(logit))))
        std = tf.stop_gradient(tf.sqrt(variance + 1e-7))
        #tf.print(f'std_nose:{tf.reduce_any(tf.math.is_nan(std))}|mean:{tf.reduce_mean(std)},max:{tf.reduce_min(std)}, min:{tf.reduce_max(std)}',end='|')

        #k= 1e-5 + (0.03 - 1e-5) *std
        k= tf.stop_gradient(tf.minimum(std*0.01,0.03))
        #tf.print('k:',k)
        nose=tf.random.normal(tf.shape(logit),0.0,k)
        ##tf.print('nose is add in logit')#--------------------------------------------
        return tf.stop_gradient(nose)
    
    
    def _ffa(self, x, expert_id, expert_prob, training=None):

        num_tokens = tf.shape(x)[0]
        d_model = tf.shape(x)[-1]

        output = tf.zeros((num_tokens, d_model), dtype=x.dtype)

        for e in range(self.num_expert):

            mask = tf.equal(expert_id, e)
            indices = tf.where(mask)
            idx = indices[:, 0]

            x_e = tf.gather(x, idx)
            prob_e = tf.gather(expert_prob, idx)

            w1 = self.expert.w1[e]
            b1 = self.expert.b1[e]
            w2 = self.expert.w2[e]
            b2 = self.expert.b2[e]

            y = tf.matmul(x_e, w1) + b1
            y = keras.activations.gelu(y)
            y = self.dropont(y, training=training)
            y = tf.matmul(y, w2) + b2

            y *= tf.expand_dims(prob_e, -1)

            output = tf.tensor_scatter_nd_add(
                output,
                tf.expand_dims(idx, 1),
                y
            )

        return output




    
    def _loss(self,gates,num_token):
        # не дать роутеру бездумно распределять ( насколько роутер уверен)
        entropy_per_token=-tf.reduce_sum(gates*tf.math.log(gates+1e-5),-1)
        entropy=tf.reduce_mean(entropy_per_token)
        loss_1=(entropy/tf.math.log(tf.cast(self.num_expert,tf.float32)))

        #важность экспертов( как средняя вероятность отличается от нормального)
        importance = tf.reduce_sum(gates, axis=[0,1])
        target = tf.cast(num_token, tf.float32) * self.top_k / self.num_expert
        loss_2 = tf.reduce_mean(((importance - target) / (target+1e-5)) ** 2)

        return (loss_1+loss_2)/self.l,loss_1,loss_2
    
    @property
    def metrics(self):
        return [self.metric1,self.metric2]
    
    def compute_output_shape(self, input_shape):
        # input_shape = (batch, seq, d_model)
        return tf.TensorShape(input_shape)  # точно такая же форма, как input


    def get_config(self):
        base_conf = super().get_config()
        config={
            'd_ff':self.d_ff,
            'num':self.num,
            'num_expert':self.num_expert,
            'top_k':self.top_k,
            'l':self.l
        }
        #d_ff,num=0,num_expert=8,top_k=2,l=3
        return {**base_conf,**config}
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
     
       



@keras.saving.register_keras_serializable('position_init')
class position_initial(keras.initializers.Initializer):
    def __init__(self,T,**kwargs):
        super().__init__(**kwargs)
        self.T=T

    def __call__(self, shapes,dtype=None,**kwargs):
        if shapes!=(self.T,self.T):
            raise ValueError('dont corretc shape in position init')
        
        p = tf.range(0, self.T, 1,dtype=dtype or tf.int32)
        delta = (p[None,:] - p[:,None]) + (self.T - 1)
        return delta
    
    def get_config(self):
        conf=super().get_config()
        new_conf={'T':self.T}
        return {**new_conf,**conf}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)




@keras.saving.register_keras_serializable('castom_MHA')
class castom_MHA(layers.MultiHeadAttention):
    def __init__(self, num_heads, key_dim, value_dim=None, dropout=0, use_bias=True, output_shape=None, attention_axes=None, flash_attention=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, **kwargs):
        super().__init__(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, **kwargs)
    
    def build(self, query_shape, value_shape, key_shape=None):
        self.initialize_pos(query_shape)
        return super().build(query_shape, value_shape, key_shape)
    

    def initialize_pos(self,query_shape):
        if len(query_shape) != 3:
            raise Exception(f'don`t correct shape you:{query_shape} | need: (batch,seq,dir_per_head)')
        
        self.T = query_shape[-2]
        self.L = 2*self.T - 1
        self.weight_pos = self.add_weight((self.num_heads,self.L),dtype = tf.float32,initializer = tf.random_normal_initializer(stddev=0.02),trainable=True,name='position')
        
        self.pos_init=position_initial(self.T)
        self._ind =self.add_weight((self.T,self.T),dtype=tf.int64,trainable=False,initializer=self.pos_init,name='const_pos')


    @property
    def pos_enoding(self):
        pos=tf.gather(self.weight_pos,self._ind,axis=1,batch_dims=0)
        return pos[tf.newaxis,...]
    

    def _compute_attention(self, query, key, value, attention_mask=None, training=None, return_attention_scores=False):
        # Check for flash attention constraints
        if self._flash_attention and return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
            self._dropout > 0.0
            or return_attention_scores
            or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                    len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = ops.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = ops.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=self.pos_enoding,
                mask=attention_mask,
                scale=self._inverse_sqrt_key_dim,
                is_causal=False,
                flash_attention=False,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        attention_scores+=self.pos_enoding
        # Apply the mask using the custom masked softmax
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # Apply dropout to the attention scores if needed
        if self._dropout > 0.0:
            final_attn_scores = self._dropout_layer(
                attention_scores, training=training
            )
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = ops.einsum(
            self._combine_equation, final_attn_scores, value
        )
        return attention_output, attention_scores
    
    def get_config(self):
        conf=super().get_config()
        return conf
    @classmethod
    def from_config(cls,config):
        return cls(**config)
        



@keras.saving.register_keras_serializable('Transformer_block')
class Transformer_block(layers.Layer):
    def __init__(self,num_head,key_dim,d_ff,num,value_dim=None,dropout=0.1,l=None,
                  num_experts=1,top_k=1,*, activity_regularizer=None, trainable=True, dtype=None, autocast=True,
                    name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.mha=castom_MHA(num_head,key_dim,value_dim,dropout=dropout,activity_regularizer=keras.activations.gelu,kernel_regularizer=keras.regularizers.l2(1e-4))
        self.moe=Moe(d_ff,num,num_experts,top_k,l)
        
    def build(self, input_shape):
        self.norm0=RMSNorm(input_shape[-1])
        self.norm1=RMSNorm(input_shape[-1])
        return super().build(input_shape)
    
    def call(self,x,training=None,step=None):
        x=self.norm0(x)
        attention=self.mha(x,x,training=training)
        x=attention+x

        x=self.norm1(x)
        y=self.moe(x,training=training,step=step)
        y = tf.ensure_shape(y, x.shape)
        y=x+y
        return y
    
    def get_config(self):
        conf=super().get_config()
        conf.update({
        'num_head': self.mha.num_heads,
        'key_dim': self.mha.key_dim,
        'value_dim': self.mha.value_dim,
        'dropout': self.mha.dropout,
        'd_ff': self.moe.d_ff,
        'num': self.moe.num,
        'num_experts': self.moe.num_expert,
        'top_k': self.moe.top_k,
        'l': self.moe.l,
    })
        return conf
    
    @classmethod
    def from_config(cls,conf):
        return cls(**conf)

# выходной слой для непрерывной политики
@keras.saving.register_keras_serializable('output_cout')
class Output_cout(layers.Layer):
    def __init__(self,num,*argv):
        super().__init__(*argv)
        self.mean=layers.Dense(num)
        self.num=num
        self.std=self.add_weight(shape=(num,),
                                 name='means',
                                 initializer=keras.initializers.Constant(-0.5),
                                 dtype=tf.float32,
                                 trainable=True)
    def call(self,x):
        return self.mean(x),tf.math.exp(tf.clip_by_value(self.std,-4,-0.55))
    def get_config(self):
        conf=super().get_config()
        conf.update({'num':self.num})
        return conf
    
    @classmethod
    def from_config(cls,conf):
        return cls(conf)
    
if __name__ =='__main__':
    print(RMSNorm(1)(tf.constant([1.,2.,3.,4.,5.,6.,7.])))
