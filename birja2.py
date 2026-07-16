import tensorflow as tf
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import datetime
import ta
import pandas_ta 
from time import sleep
import parser_data   
#----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------


class Trading_3():
    def __init__(self,render_mode=None,other_param=2,balance=100):
        super().__init__()
        #open	-
        # high	-
        # low	-
        # close-	
        # volue	-
        # rsi   -
        # adx  -
        # bb  -
        # obv	-
        # atr -
        # position_size
        #unrealized_pnl
        #time_in_trade  
        #direction
        #sin_hour = sin(2 * π * hour / 24) -
        #cos_hour = cos(2 * π * hour / 24) -
        #sin_day = sin(2 * π * day / 7)  -
        #cos_day = cos(2 * π * day / 7)  -
        '''open, high, low, close	z-score по окну 50	±3
        volume	log1p + z-score	±3
        RSI, ADX	z-score (если не [0–1])	±3
        BB_%b	[0–1], без z-score	❌
        OBV	diff() + z-score	±3
        ATR	log1p + z-score	±3
        unrealized_pnl	clip ±50% → z-score	±3
        position_size, time_in_trade, direction	[0–1] или one-hot	❌
        sin_hour, cos_hour, sin_day, cos_day	уже [-1,1]	❌'''
        
        self.unrealized_pnl=0
        self.time_in_trade=0
        self.direction=0
        self.prise=0
        self.trade_count=0
        self.step_per_episode=96*7
        self.step_in_episode=0
        
        self.pars=parser_data.parse()
        self.action_space={'scalars':3,'level':3} # 3 действия 3 уровня 
        self.observation_space ={
            'window_data': [(150,9),(100,9),(20,9)],
            'global': (4), # начало 15м, начало 4h, начало 1w, последняя на 15м
            'action_one_hot': 4,
            'scalars':5,
            'level':3,
            "un_pnl":1,
            'time_in_trade':1
        }
        
        
        # path1=Path("from_3_4.parquet")  # 15m
        # path2=Path("from_2_4.parquet") # 4h
        # path3=Path("from_1_4.parquet") # 1w
        # #path1=Path("/content/drive/MyDrive/Colab Notebooks/from_3_4.parquet")  # 15m
        # #path2=Path("/content/drive/MyDrive/Colab Notebooks/from_2_4.parquet") # 4h
        # #path3=Path("/content/drive/MyDrive/Colab Notebooks/from_1_4.parquet") # 1w
        # self.data_15=pd.read_parquet(path1)
        # self.data_4=pd.read_parquet(path2)
        # self.data_1=pd.read_parquet(path3)
        t=datetime.datetime.now(datetime.timezone.utc)
        dat,iz=self.pars(datetime.datetime(year=t.year,month=t.month,day=t.day,hour=t.hour,minute=15*(t.minute//15),tzinfo=datetime.timezone.utc)-datetime.timedelta(minutes=1))
        self.data_15=dat['15m']
        self.data_4=dat['4h']
        self.data_1=dat['1w']
        self.window_15=150
        self.window_4=99
        self.window_1=19

        self.balance=float(balance)
        self.balance_start=balance
        self.risk_per_trade=self.balance*0.01

        #self.start2=datetime.datetime.fromisoformat('2026-05-17 00:00:00+00:00')
        self.start2=self.data_1['timestamp'].iloc[-1]-datetime.timedelta(days=1)

        for i in range(len(self.data_4['timestamp'])):
            if self.start2==self.data_4['timestamp'].iloc[i]:
                self.index2=i
                break

        # self.start2=self.data_4['timestamp'].iloc[self.window_4]
        # self.index2=self.window_4

        for i in range(len(self.data_15['timestamp'])):
            if self.start2+datetime.timedelta(hours=3,minutes=46)<=self.data_15['timestamp'].iloc[i]:
                self.index1=i
                self.start1=self.data_15['timestamp'].iloc[i]
                break

        for i in range(len(self.data_1['timestamp'])):
            if self.start2<self.data_1['timestamp'].iloc[i]:
                self.index3=i-2
                self.start3=self.data_1['timestamp'].iloc[i-2]
                break

        print('start2:',self.start2,'| index2:',self.index2,'| start2(for index):',self.data_4['timestamp'].iloc[self.index2],'end:',self.data_4['timestamp'].iloc[-1])
        print('start1:',self.start1,'| index1:',self.index1,'| start1(for index):',self.data_15['timestamp'].iloc[self.index1],'end:',self.data_15['timestamp'].iloc[-1])
        print('start3:',self.start3,'| index3:',self.index3,'| start3(for index):',self.data_1['timestamp'].iloc[self.index3],'end:',self.data_1['timestamp'].iloc[-1])

        self.render_mode=render_mode
        self.len_data=len(self.data_15)-1

        self.stop=0
        self.take=0
        self.open=0
        self.trades=0
        self.open_k=False
        self.risk_k=1
        self.start_load=0
        self.index_data=[self.index1,self.index2,self.index3]
        self.start_data=[self.start1,self.start2,self.start3]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
    
        info={'balance':self.balance}

        self.balance=self.balance_start
        self.unrealized_pnl=0
        self.time_in_trade=0
        self.direction=0
        self.trades=0

        self.prise=0
        self.trade_count=0
        self.stop=0
        self.take=0
        self.open=0
        self.open_k=False
        self.risk_k=1
        self.start_load=0
        self.step_in_episode=0
        self.index1,self.index2,self.index3=self.index_data
        self.start1,self.start2,self.start3=self.start_data

        observation,tf_4h,tf_1w=self._get_observation()
        info['4h']=tf_4h
        info['1w']=tf_1w
        return observation,info
    
    def get_index(self,_observation)->dict:
        """
        Docstring для get_index
        возвращает индексы и данные для данных которые прямо сейчас вернула среда
        -> index1,index2,index3
        """
        # save: action_one_hot,level,time_in_trade,un_pnl
        _observation=_observation[0]
        return {'index1':tf.constant(self.index1,tf.int32).numpy().tolist(),
                'index2':tf.constant(self.index2,tf.int32).numpy().tolist(),
                'index3':tf.constant(self.index3,tf.int32).numpy().tolist(),
                'action_one_hot':tf.constant(_observation['action_one_hot'],tf.float32).numpy().tolist(),
                'level':tf.constant(_observation['level'],tf.float32).numpy().tolist(),
                'time_in_trade':tf.constant(_observation['time_in_trade'],tf.float32).numpy().tolist(),
                'un_pnl':tf.constant(_observation['un_pnl'],tf.float32).numpy().tolist(),
                }
    
    def get_obsevation_for_index(self,_observation):
        _ind1=self.index1
        _ind2=self.index2
        _ind3=self.index3
        _st1=self.start1
        _st2=self.start2
        _st3=self.start3

        self.index1=int(_observation['index1'])
        self.index2=int(_observation['index2'])
        self.index3=int(_observation['index3'])
        self.start1=self.data_15['timestamp'].iloc[int(_observation['index1'])]
        self.start2=self.data_4['timestamp'].iloc[int(_observation['index2'])]
        self.start3=self.data_1['timestamp'].iloc[int(_observation['index3'])]

        _o,tf4h,tf1w=self._get_observation()
        _o['action_one_hot']=_observation['action_one_hot']
        _o['level']=_observation['level']
        _o['time_in_trade']=_observation['time_in_trade']
        _o['un_pnl']=_observation['un_pnl']
        # self.index1=_ind1
        # self.index2=_ind2
        # self.index3=_ind3
        # self.start1=_st1
        # self.start2=_st2
        # self.start3=_st3

        return _o,{'4h':tf.constant(tf4h,tf.bool),'1w':tf.constant(tf1w,tf.bool)}

    #'timestamp','open','high','low','close','volue',rsi,	adx,	bb,	obv,atr
    
        """Никита, [19.11.2025 7:01]
    исправь комисию на 0.035% для тэйкера и 0.01 для мэйкера и еще исправь чтобы вход в сделку было всегда от 1 процента

    Никита, [27.11.2025 18:46]
    Самому делать свечу из младшего тф

    Никита, [27.11.2025 18:46]
    Увеличить диапазон цены т. К. Старший тф

    Никита, [27.11.2025 18:46]
    Одинаковая цена

    Никита, [03.12.2025 16:57]
    попробовать поменять немного модель и ее глубену через блоки внимания 

    починитть всю среду и только когда попробую верхнее то добавить другие ТФ и"""
    def _get_observation(self):
        
        start0=self.index1-self.window_15
        start1=self.index2-self.window_4
        start2=self.index3-self.window_1
        end0=self.index1
        end1=self.index2
        end2=self.index3
        tf_4h=False
        tf_1w=False
        #print(self.data_15['timestamp'].iloc[end0])
        #print('---------------------------------------------')
        # print('h4 begin:',self.data_4['timestamp'].iloc[end1+1],"| end:",self.data_4['timestamp'].iloc[end1+1]+datetime.timedelta(hours=4))
        if not(self.data_4['timestamp'].iloc[end1+1] <= self.data_15['timestamp'].iloc[self.index1] < self.data_4['timestamp'].iloc[end1+1]+datetime.timedelta(hours=4)):
            # print("4h+1")
            self.index2+=1
            end1+=1
            start1+=1
            self.start2+=datetime.timedelta(hours=4)
            tf_4h=True
        # print('1w begin:',self.data_1['timestamp'].iloc[end2+1],"| end:",self.data_1['timestamp'].iloc[end2+1]+datetime.timedelta(hours=24*7))
        
        if not(self.data_1['timestamp'].iloc[end2+1] <= self.data_15['timestamp'].iloc[self.index1] < self.data_1['timestamp'].iloc[end2+1]+datetime.timedelta(days=7)):
            # print('1w +1')
            tf_1w=True
            self.index3+=1
            end2+=1
            start2+=1
            self.start3+=datetime.timedelta(days=7)


        #np.set_printoptions(threshold=np.inf)
        datas={'15m':None,'4h':None,'1w':None}
        for start,end,tf,data,k in zip((start0,start1,start2),(end0,end1,end2),('15m','4h','1w'),(self.data_15,self.data_4,self.data_1),(100,3,1)):
            # print(f'start:{data['timestamp'].iloc[start+1]} | end:{data["timestamp"].iloc[end]} | len:{len(np.array(data['open'].iloc[start+1:end+1].to_list(),np.float32))}')
            open=np.array(data['open'].iloc[start+1:end+1].to_list(),np.float32)
            close=np.array(data['close'].iloc[start+1:end+1].to_list(),np.float32)
            high=np.array(data['high'].iloc[start+1:end+1].to_list(),np.float32)
            low=np.array(data['low'].iloc[start+1:end+1].to_list(),np.float32)
            volue=np.array(data['volue'].iloc[start+1:end+1].to_list(),np.float32)
            rsi=np.array(data['rsi'].iloc[start+1:end+1].to_list(),np.float32)
            bb=np.array(data['bb'].iloc[start+1:end+1].to_list(),np.float32)
            adx=np.array(data['adx'].iloc[start+1:end+1].to_list(),np.float32)
            atr=np.array(data['atr'].iloc[start+1:end+1].to_list(),np.float32)
            #print(f"{tf};{data['timestamp'].iloc[end]}")
            #if tf=='15m':
             #   print('low1:',low,'| hig1h:',high,'| close1:',close)

            if tf=='4h':
                for i in range(self.index1,self.index1-17,-1):
                    if self.data_4['timestamp'].iloc[self.index2]+datetime.timedelta(hours=4)>=self.data_15['timestamp'].iloc[i]:
                        candel=self.data_15.iloc[i:self.index1+1]
                        t=self.data_4['timestamp'].iloc[self.index2+1]
                        #print(t)
                        h=candel['high'].max()
                        l=candel['low'].min()
                        o=candel['open'].iloc[0]
                        c=candel['close'].iloc[-1]
                        v=candel['volue'].sum()
                        d=self.data_4.iloc[self.index2-30:self.index2+1,:6]
                        d[['close','open','high','low','volue']]=d[['close','open','high','low','volue']].astype(np.float32)
                        d=pd.DataFrame(self.data_4.iloc[self.index2-30:self.index2+1,:6],columns=['timestamp','open','high','low','close','volue'])

                        d=pd.concat([d,pd.DataFrame(np.reshape([t,o,h,l,c,v],(1,6)),columns=d.columns)],ignore_index=True)
                        d[['close','open','high','low','volue']]=d[['close','open','high','low','volue']].astype(np.float32)
                        rsi1=ta.momentum.rsi(d['close'],14)
                        '''print('high:',h)
                        print('low:',l)
                        print('close:',c)'''
                        adx1=pandas_ta.adx(d['high'],d['low'],d['close'],14)
                        bb1=ta.volatility.BollingerBands(d['close'],20,2)
                        bolban1=bb1.bollinger_pband()
                        #print(d.iloc[-2:])
                        atr1=ta.volatility.average_true_range(d['high'],d['low'],d['close'])
                        #d=pd.DataFrame(np.reshape([t,o,h,l,c,v,rsi1.to_list()[-1],adx1.to_list()[-1],bolban1.to_list()[-1],atr1.to_list()[-1]],(1,10)),columns=self.data_4.columns)
                        #print(d)
                        open=np.append(open,[o],0)
                        close=np.append(close,[c],0)
                        high=np.append(high,[h],0)
                        low=np.append(low,[l],0)
                        volue=np.append(volue,[v],0)
                        rsi=np.append(rsi,[rsi1.to_list()[-1]],0)

                        adx=np.append(adx,[adx1['ADX_14'].iloc[-1]],0)
                        bb=np.append(bb,[bolban1.to_list()[-1]],0)
                        atr=np.append(atr,[atr1.to_list()[-1]],0)
                        '''print('open:',len(open))
                        print('high:',len(high))
                        print('low:',len(low))
                        print('close:',len(close))
                        print('volue:',len(volue))
                        print('rsi:',len(rsi))
                        print('adx:',len(adx))
                        print('bb:',len(bb))
                        print('atr:',len(atr))'''

                        # print('pred0:',self.data_4['volue'].iloc[self.index2+1])
                        # print('pred1:',self.data_4['volue'].iloc[self.index2])
                        # print('v:',v)
                        # print(len(candel))
                        # print('my_4h:',volue[-1:])
                        #for p,f in zip(volue[-100:],self.data_4['timestamp'].iloc[self.index2-99:self.index2+1]):
                         #   print('volue:',p,' time(4h):',f)
                        break
            if tf=='1w':
                #print('time 1 candel id 1w:',self.data_1['timestamp'].iloc[self.index3])
                len_data=min(672,self.index1)
                #print('len:',len(close),'now data:',self.data_1['timestamp'].iloc[self.index3]+datetime.timedelta(days=7), '| end data:',self.data_15['timestamp'].iloc[self.index1],'start data:',self.data_15['timestamp'].iloc[self.index1-len_data],'| len_data:',len_data,'| len:',self.index1)
                for i in range(self.index1,self.index1-len_data-1,-1):
                    if self.data_1['timestamp'].iloc[self.index3]+datetime.timedelta(days=7)>=self.data_15['timestamp'].iloc[i] or i==0:
                        candel=self.data_15.iloc[i:self.index1+1]
                        t=self.data_1['timestamp'].iloc[self.index3+1]
                        #print(t)
                        #print('свечи на 15м')
                        #print(candel)
                        
                        #print('4h')
                        #print('pred:',self.data_1['timestamp'].iloc[self.index3])
                        h=candel['high'].max()
                        l=candel['low'].min()
                        o=candel['open'].iloc[0]
                        #print(len(candel))
                        c=candel['close'].iloc[-1]
                        v=candel['volue'].sum()
                        #print(candel['volue'].mean())
                        #print('v:',v)
                        #print("mean v:",self.data_1['volue'].iloc[self.index3-30:self.index3+1].mean())
                        # d=self.data_1.iloc[self.index3-30:self.index3+1,:6]
                        # d[['close','open','high','low','volue']]=d[['close','open','high','low','volue']].astype(np.float32)
                        d=pd.DataFrame(self.data_1.iloc[self.index3-20:self.index3+1,:6],columns=['timestamp','open','high','low','close','volue'])
                        #print('len d-1:',len(d))

                        d=pd.concat([d,pd.DataFrame(np.reshape([t,o,h,l,c,v],(1,6)),columns=d.columns)],ignore_index=True)
                        # print('len d0:',len(d))

                        d[['close','open','high','low','volue']]=d[['close','open','high','low','volue']].astype(np.float32)
                        rsi1=ta.momentum.rsi(d['close'],14)
                        #print('len d1:',len(d))
                        '''print('high:',h)
                        print('low:',l)
                        print('close:',c)'''
                        adx1=pandas_ta.adx(d['high'],d['low'],d['close'],14)
                        bb1=ta.volatility.BollingerBands(d['close'],20,2)
                        bolban1=bb1.bollinger_pband()
                        #print(d.iloc[-2:])
                        atr1=ta.volatility.average_true_range(d['high'],d['low'],d['close'])
                        #d=pd.DataFrame(np.reshape([t,o,h,l,c,v,rsi1.to_list()[-1],adx1.to_list()[-1],bolban1.to_list()[-1],atr1.to_list()[-1]],(1,10)),columns=self.data_4.columns)
                        #print(d)

                        open=np.append(open,[o],0)
                        close=np.append(close,[c],0)
                        high=np.append(high,[h],0)
                        low=np.append(low,[l],0)
                        volue=np.append(volue,[v],0)
                        rsi=np.append(rsi,[rsi1.to_list()[-1]],0)
                        adx=np.append(adx,[adx1['ADX_14'].iloc[-1]],0)
                        bb=np.append(bb,[bolban1.to_list()[-1]],0)
                        atr=np.append(atr,[atr1.to_list()[-1]],0)
                        
                        #print('pred0:',self.data_1['volue'].iloc[self.index3+1])
                        #print('pred1:',self.data_1['volue'].iloc[self.index3])

                        #print('len 1w:',len(close))
                        #print('my_1w:',volue[-1:])
                        #for p,f in zip(volue[-30:],self.data_1['timestamp'].iloc[self.index3-29:self.index3+1]):
                         #   print('volue:',p,' time(1w):',f)
                        break
            #open=self.z_score(open)[-self.window:]
            #close=self.z_score(close)[-self.window:]
            #high=self.z_score(high)[-self.window:]
            #low=self.z_score(low)[-self.window:]
            #if tf=='15m':
                #print('real normalization:',open[0])
                #open_trade=open[0]
            #if tf=='15m':
            #    print('not normaliz data15:',high[-5:],low[-5:])
            close1=open[0]
            open=((open/close1)-1)*k
            close=((close/close1)-1)*k
            high=((high/close1)-1)*k
            low=((low/close1)-1)*k
            #if tf=='15m':

                #print('normaliz data15:',close[-15:])

            #open=open[-self.window:]
            #close=close[-self.window:]
            #high=high[-self.window:]
            #low=low[-self.window:]

            volue=np.log(volue+1)
            volue=volue/volue.mean()
            volue=np.clip(volue-0.5,-3,+3)

            rsi=rsi/100
            rsi=np.clip(rsi,0,1)
            adx=adx/100
            adx=np.clip(adx,0,1)
            adx=adx
            rsi=rsi

            atr=np.log(atr+1)
            mean=np.mean(atr)
            atr=np.clip((atr-mean)/atr.std()+1e-8,-3,+3)
            '''print('open:',len(open), "type:",type(open))
            print('high:',len(high), "type:",type(high))
            print('low:',len(low), "type:",type(low))
            print('close:',len(close), "type:",type(close))
            print('volue:',len(volue), "type:",type(volue))
            print('rsi:',len(rsi), "type:",type(rsi))
            print('adx:',len(adx), "type:",type(adx))
            print('bb:',len(bb), "type:",type(bb))
            print('atr:',len(atr), "type:",type(atr))'''

            colum=np.column_stack([open,high,low,close,volue,rsi,adx,bb,atr])
            datas[tf]=colum


        #print('end:',self.data_15['timestamp'].iloc[end0],' index:',end0)
        #print('end 4h:',self.data_4['timestamp'].iloc[end1],' index:',end1)
        #print('start _p:',self.data_15['timestamp'].iloc[start0+1:end0+1].tolist()[0],' end:',self.data_15['timestamp'].iloc[start0+1:end0+1].tolist()[-1])

        time=self.data_15['timestamp'].iloc[end0]


        hour=2*np.pi*time.hour/24
        day=2*np.pi*time.day_of_week/7
        
        #un_pnl=np.clip(self.unrealized_pnl*self.risk_k,-1,1)/0.5
        un_pnl=np.clip(self.unrealized_pnl*self.risk_k*0.8,-1,5)
        time_in_t=np.tanh(self.time_in_trade/50)
        discret=3 if not self.open_k and self.prise!=0 else self.direction+1
        
        dirat=np.eye(4)[discret]
        #print('disret:',dirat)
        open_trade=self.data_15['open'].iloc[self.index1-self.window_15+1]
        open1=((((self.open+1)*self.start_load)/open_trade)-1)*100 if not self.open_k else ((self.prise/open_trade)-1)*100
        #print('normalization open0:',open_trade)
        #print('normalization open1:',self.data_15['open'].iloc[self.index1-self.window_15+1])
        take1=((((self.take+1)*self.start_load)/open_trade)-1)*100
        #reverse normaliz: ((take1/100+1)*open_trade) /self.prise-1
        stop1=((((self.stop+1)*self.start_load)/open_trade)-1)*100
        #print('my take:',(self.take+1)*self.prise,'| my stop:',((self.stop+1)*self.prise),'| open:',((self.open+1)*self.prise),"| open 1:",(self.prise))
        # close1=open[0]
        #     open=((open/close1)-1)*k
        #     close=((close/close1)-1)*k
        #     high=((high/close1)-1)*k
        level=[open1,take1,stop1] if self.prise!=0 else [0,0,0]
        #print('leve(!) open:',self.open,'| take:',self.take,'| stop:',self.stop,'| diration:',self.direction)
        #print('level:',level)

        sin_hour, cos_hour, sin_day, cos_day=np.sin(hour),np.cos(hour),np.sin(day),np.cos(day)
        self.time=time
        # print('close:',self.data['close'][end-1])
        scalar=np.array([sin_hour,cos_hour,sin_day,cos_day],np.float32)
        global_level=np.array([self.data_15['open'].iloc[start0+1],self.data_4['open'].iloc[start1+1],self.data_1['open'].iloc[start2+1],self.data_15['close'].iloc[end0]])
        #print('scal:',scalar,'| balanse:',self.balance)

        #print('15m:',self.data_15['timestamp'].iloc[self.index1])
        #print('4h:',self.data_4['timestamp'].iloc[self.index2])
        #print('1w:',self.data_1['timestamp'].iloc[self.index3])
        
        global_level=np.log(1+np.exp(global_level/10000))
        level=np.array(level,np.float32)


        # save: action_one_hot,level,time_in_trade,un_pnl
        return {'window_data':datas,"action_one_hot":dirat,"scalars":scalar,"global":global_level,'level':level,'un_pnl':un_pnl,'time_in_trade':time_in_t},tf_4h,tf_1w
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # добавь данные в скаляры уровень входа и стоп с тэйком
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def trader(self):
         #проверка и реализация до действий но для данных обновленных на 1
        pr=0
        bal1=self.balance
        high=float(self.data_15["high"].iloc[self.index1])
        low=float(self.data_15["low"].iloc[self.index1])
        reward=0
        close=float(self.data_15["close"].iloc[self.index1])
        #print('low:',low,'| high:',high,'| close:',close)
        if  not self.open_k and self.prise!=0 and self.direction!=0:
            #print('load')
            #print('stop:',self.stop,'| take:',self.take,"| open:",self.open,'| nor o:',self.prise*(self.open+1))

            if low<=self.prise*(self.open+1)<=high:
                reward+=0.1
                #print('YES!!!')
                self.prise=self.prise*(self.open+1)
                self.unrealized_pnl=0
                self.trade_count+=1
                self.open_k=True
                pr+=1
            else:
                reward-=0

        high=float(((high-self.start_load)/self.start_load)*self.direction if self.prise!=0 else 0)
        low=float(((low-self.start_load)/self.start_load)*self.direction if self.prise!=0 else 0)
        #print('nor hig:',high,'| nor low:',low)
        #nor hig: -0.0010571636123103238 | nor low: 0.0018928544008126147
        #or hig: -0.0014870210492881459 | nor low: 0.0029134062887997414
        #nor hig: -0.0003181544726733968 | nor low: 0.003308806515803616
        if (low*self.risk_k<=self.stop*self.direction or high*self.risk_k<=self.stop*self.direction) and self.open_k:
            #reward-=self.risk_per_trade*self.risk_k*4+ self.risk_per_trade*self.risk_k*0.00045 #////////
            #print('stop!')
            #self.balance-=self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045
            if pr==0: 
                #reward-=self.risk_per_trade*self.risk_k*1+ self.risk_per_trade*self.risk_k*0.00045 #////////

                self.balance+=-1*self.risk_per_trade*self.risk_k- self.risk_per_trade*self.risk_k*0.00045 

                print(f'balanse:{-1*self.risk_per_trade*self.risk_k- self.risk_per_trade*self.risk_k*0.00045 }|take:{self.take}|stop:{self.stop}|dir:{self.direction}')
            else:
                reward-=2
                print('AAAAAARRR!!!',f'|pr:{pr}')
            #print('volue balanse:',self.risk_per_trade,'| delta balanse:',-1*self.risk_per_trade*self.risk_k- self.risk_per_trade*self.risk_k*0.00045)
            self.prise=0
            self.direction=0
            self.unrealized_pnl=0
            self.time_in_trade=0
            self.trades+=1
            
            self.stop=0
            self.take=0
            self.open=0
            self.open_k=False
            self.start_load=0
            pr+=1

            #------------
        if (high*self.risk_k>=self.take*self.direction or low*self.risk_k>=self.take*self.direction) and self.open_k:
            if pr==0: 
                reward+=1 +16*abs(self.take/self.stop)*self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045#////////

                #print('take!')
                self.trades+=1
                #self.balance-=self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045 
                self.balance+=abs(self.take/self.stop)*self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045
                print(f'balanse:{abs(self.take/self.stop)*self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045}|take:{self.take}|stop:{self.stop}|dir:{self.direction}')
            else:
                reward-=2
                print('AAAAAARRR!!!',f'|pr:{pr}')
            #print('volue balanse:',self.risk_per_trade,'| delta balanse:',abs(self.take/self.stop)*self.risk_per_trade- self.risk_per_trade*self.risk_k*0.00045)
            self.prise=0
            self.direction=0
            self.unrealized_pnl=0
            self.time_in_trade=0
            self.stop=0
            self.take=0
            self.open=0
            self.open_k=False
            self.start_load=0
            pr+=1
            #------------

        if pr>2:
            print('AAAAAARRR!!!')
            reward+=-8
            self.balance=bal1


        if self.prise!=0 and self.open_k and pr==0 :
            self.time_in_trade+=1
            d_p_l=float(((close-self.prise)/self.prise)*self.direction if self.prise!=0 else 0)
            #print('d_p_l in trade:',d_p_l*100,'|close:',close,'| self.prise:',self.prise)
            self.unrealized_pnl=d_p_l*self.risk_k*50
            rew=d_p_l if d_p_l>0 else d_p_l*0
            reward+=rew+(-1*np.tanh(self.time_in_trade/70)/3)

        return float(reward)
        
    def _trade(self,action,stop,take,open):
        """0 — ничего не делать (держать)
        1 — открыть/удерживать 
        2 — закрыть позицию
        """
        open/=100
        reward=float(0.0)
        close=float(self.data_15["close"].iloc[self.index1])
        #d_p_l=float(((close-self.prise)/self.prise)*self.direction if self.prise!=0 else 0)
        #strong=float(-0.2 if (rsi>70 or rsi<30 or adx>25 ) and self.direction==0 else 0)
        #print('d_p_l in _trade:',d_p_l,'|close:',close,'| self.prise:',self.prise)
        open_trad=self.data_15['open'].iloc[self.index1-self.window_15+1]
        stop=((stop/100+1 if stop/100!=np.inf else 1.0)*open_trad)/close-1.0
        take=((take/100+1 if take/100!=np.inf else 1.0)*open_trad)/close-1.0

        #print('leve(2) open:',open,'| take:',take,'| stop:',stop)

        
        
        '''self.current_step=self.narmol-self.window
        self.balance=self.balance_start
        self.position=0
        self.unrealized_pnl=0
        self.time_in_trade=0
        self.direction=0'''

        #print('action:',action)
        if action==0 and self.open_k and self.prise!=0:
            pass
            #print('action=0,prise!=0')
            #reward+=d_p_l*10
            #reward+= 0.2 if d_p_l>0 else 0
            #reward+= -0.3 if d_p_l<0 else 0
        elif action==0 and self.prise!=0 and not self.open_k:
            # print('load')
            # print('stop:',self.stop,'| take:',self.take,"| open:",self.open)

            # if low<=self.open<=high:
            #     self.prise=self.prise*(self.open+1)
            #     self.unrealized_pnl=0
            #     self.trade_count+=1
            #     self.open_k=True
            pass

        elif action==1 and self.open_k and self.prise!=0:
            diract=1 if take>0 else -1 if take<0 else 0
            if self.direction==diract or diract==0:
                #print('action=1,prise!=0, diration=self.dir')
                #reward+=d_p_l*10
                #reward+= 0.2 if d_p_l>0 else 0
                #reward+= -0.3 if d_p_l<0 else 0 
                pass
            else:

                if take == stop or ((open<stop and open<take) or (open>take and open>stop)):
                    #print('dont corrate1')
                    #print('action=1,prise!=0, diration=self.dir')

                    reward-=0 # то же что и при обычном ожидании 
                    return float(reward)

                #print('action=1,prise!=0, diration!=self.dir')
                # if d_p_l>0:
                #     reward+=d_p_l*12*10
                # else:
                #     reward+=d_p_l*12*5
                # reward-=0.015
                # reward+=-np.clip(self.trade_count-150,0,1000)/190*3
                
                #reward-=np.clip(40-self.trade_count,0,40)/100
                #print('prise:',self.prise)
                close_per_stop=float(((close-self.prise)/self.prise)*self.direction if self.prise!=0 else 0)
                if self.prise!=0:
                    delta= close_per_stop/abs(self.stop)
                    self.trades+=1
                    self.balance+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k
                    print(f'delta:{delta}|balanse:{(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k}|take:{self.take}|stop:{self.stop}|dir:{self.direction}')
                reward+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k)*(16 if delta>0 else 0 if delta<0 else 0) -0.0007*self.risk_per_trade*self.risk_k
                reward-=0.
                    #print('close_per_stop:',close_per_stop,'| delta:',delta,'|delta balanse:',delta*self.risk_per_trade*self.risk_k -0.0007*self.risk_per_trade*self.risk_k)
                if self.balance>=self.risk_per_trade:
                    
                    self.open=open
                    self.take=take
                    self.stop=stop
                    self.prise=close
                    self.time_in_trade=0
                    self.direction=diract
                    self.unrealized_pnl=0
                    self.open_k=False
                    self.start_load=close
                else:
                    self.open=0
                    self.take=0
                    self.stop=0
                    self.prise=0
                    self.time_in_trade=0
                    self.direction=0
                    self.unrealized_pnl=0
                    self.open_k=False
                    self.start_load=0
                    reward-=0.

        elif action==1 and self.prise!=0 and not self.open_k:
            # print('load')
            # print('stop:',self.stop,'| take:',self.take,"| open:",self.open)
            # if low<=self.open<=high:
            #     self.prise=self.prise*(self.open+1)
            #     self.unrealized_pnl=0
            #     self.trade_count+=1
            #     self.open_k=True
            pass

        elif action==1 and self.prise==0 and self.balance>=self.risk_per_trade:
            #print('start load')

            if take == stop or ((open<stop and open<take) or (open>take and open>stop)):
                #print('dont corrate')
                reward-=0
                return float(reward)
                

            diract=1 if take>0 else -1 if take<0 else 0
            #print('leve(1) open:',open,'| take:',take,'| stop:',stop,'| diration:',diract)
            reward+=0.1
            self.open=open
            self.take=take
            self.stop=stop
            self.time_in_trade=0
            self.prise=close
            self.direction=diract
            self.unrealized_pnl=0
            self.open_k=False
            self.start_load=close

        elif action==2 and self.prise!=0:
            #------------
            #print('close!')
            # rt=1

            # if d_p_l>0:
            #     reward+=d_p_l*12*10
            # else:
            #     reward+=d_p_l*12*5
            # reward-=0.015
            # reward+=-np.clip(self.trade_count-150,0,1000)/190*3
            #reward-=np.clip(40-self.trade_count,0,40)/100

            #print('prise:',self.prise)
            if self.prise!=0 and self.open_k:
                #print('TRADE END!!!')
                close_per_stop=float(((close-self.prise)/self.prise)*self.direction if self.prise!=0 else 0)
                delta= close_per_stop/abs(self.stop)
                
                self.trades+=1
                self.balance+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k
                print(f'delta:{delta}|balanse:{(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k}|take:{self.take}|stop:{self.stop}|dir:{self.direction}')
                
                #print('close_per_stop:',close_per_stop,'| delta:',delta,'|delta balanse:',delta*self.risk_per_trade*self.risk_k -0.0007*self.risk_per_trade*self.risk_k)
                reward+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k)*(16 if delta>0 else 0 if delta<0 else 0) -0.0007*self.risk_per_trade*self.risk_k-0.1
            if self.prise!=0 and not self.open_k:
                reward-=0
            self.open=0
            self.take=0
            self.stop=0
            self.prise=0
            self.time_in_trade=0
            self.direction=0
            self.unrealized_pnl=0
            self.open_k=False
            self.start_load=0
        else:
            #print('nating')
            reward+=-0.
        
        return float(reward)
    
    def end_episode(self,reward)->any:
            reward=0.0
            close=self.data_15["close"][self.index1]

            if self.prise!=0 and self.open_k:
                close_per_stop=float(((close-self.prise)/self.prise)*self.direction if self.prise!=0 else 0)
                delta= close_per_stop/abs(self.stop)
                self.trades+=1
                self.balance+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k

                print(f'delta:{delta}|balanse:{(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k) -0.0007*self.risk_per_trade*self.risk_k}|take:{self.take}|stop:{self.stop}|dir:{self.direction}')

                reward+=(-1 if delta*self.risk_per_trade*self.risk_k <=-1 else delta*self.risk_per_trade*self.risk_k)*(16 if delta>0 else 0 if delta<0 else 0) -0.0007*self.risk_per_trade*self.risk_k-0.1
            if self.prise!=0 and not self.open_k:
                reward-=0

            # total_balans=(self.balance-self.balance_start)/self.balance_start
            # if total_balans>0:
            #     reward+=total_balans*20
            # else:
            #     reward+=total_balans*8
            if (self.balance-self.balance_start)>0:
                 reward+=(self.balance-self.balance_start)*10+20
                 print('1',end=':')
            elif (self.balance-self.balance_start)<0:
                reward+=(self.balance-self.balance_start)-3
                print('2',end=':')
            else:
                reward+=-5
                print('3',end=':')


            print(f'trade: {self.trade_count}| trade_: {self.trades}| balanse: {self.balance} | reward:{reward}| edn:{self.balance-self.balance_start}')
            self.open=0
            self.take=0
            self.stop=0
            self.prise=0
            self.trades=0

            self.time_in_trade=0
            self.direction=0
            self.unrealized_pnl=0
            self.open_k=False
            self.start_load=0

            self.trade_count=0
            self.step_in_episode=0
            self.balance=self.balance_start
            return reward
    
    def step(self,action=0,take=0,stop=0,open=0): 
        
        reward=0  
        reward+=self._trade(action,stop,take,open) 
        
        self.index1+=1
        self.start1+=datetime.timedelta(minutes=15)
        self.step_in_episode+=1
        ui=False
        if self.index1>len(self.data_15)-1 or self.start1>self.data_15['timestamp'].iloc[-1]:
            print(f"{self.index1}:{self.start1}|last:{len(self.data_15)-1}:{self.data_15['timestamp'].iloc[-1]}")
            while True:
                t=datetime.datetime.now(datetime.timezone.utc)
                dat,iz=self.pars(datetime.datetime(year=t.year,month=t.month,day=t.day,hour=t.hour,minute=15*(t.minute//15),tzinfo=datetime.timezone.utc)-datetime.timedelta(minutes=1))
                if iz:
                    print(f'index1:{self.index1}|data15:{dat['15m']['timestamp'][-3:]}')
                    self.data_15=dat['15m']
                    self.data_1=dat['1w']
                    self.data_4=dat['4h']
                    break
                else:
                    if ui:
                        sleep(0.1)
                    else:
                        t=datetime.datetime.now(datetime.timezone.utc)
                        max_t=datetime.datetime(year=t.year,month=t.month,day=t.day,hour=t.hour,minute=15*(t.minute//15)+14,second=59,microsecond=999999,tzinfo=datetime.timezone.utc)
                        time_sleep=(max_t-t).total_seconds()
                        print(f'sleep:{datetime.timedelta(seconds=max(time_sleep-1,0.1))}| now:{t}| max:{max_t}|t:{t}')
                        sleep(max(time_sleep-1,0.1))
                        ui=True
        #print(f'ind:{self.index1}| time:{self.start1}')

        done= self.index1>=len(self.data_15)-1
        truncated=self.step_in_episode>=self.step_per_episode
        reward+=self.trader()
        
        #total_balans=(self.balance-self.balance_start)/self.balance_start

        if truncated :
            reward+=self.end_episode(reward)
            print(f'reward end:{reward}')
        
        # if total_balans>0:
        #     reward+=total_balans/2
        # elif total_balans<0.0:
        #     reward+=total_balans/9
        # else:
        #     reward+=-0.5
        
        observation_,tf_4h,tf_1w=self._get_observation()

        info={'balance':self.balance,'pnl':self.unrealized_pnl,'4h':tf_4h,'1w':tf_1w}
        #print(f"reward: {reward}| action: {action}| stop: {stop}| take:{take}| open: {open}")

        return observation_,reward,done,truncated,info
        # reward,rt=self._trade(action)
        # self.current_step+=1
        # self.step_in_episode+=1
        # done= self.current_step>=(self.len_data-self.window-1)
        # truncated=self.step_in_episode>=self.step_per_episode
        # info={'balance':self.balance,'position':self.position,'pnl':self.unrealized_pnl,'rt':rt}
        # #print(self.data["close"][self.current_step+self.window-1])
        # #print(self.time_in_trade)

        # observation=self._get_observation()
        # if done:
        #     self.current_step=self.narmol-self.window
        


        # return observation,reward,done,truncated,info
    