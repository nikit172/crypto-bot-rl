import requests
import json 
import ta
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from matplotlib import pyplot as plt
import time
import datetime
from datetime import timezone
import urllib
from datetime import timedelta

from urllib import parse

import concurrent.futures as fut
import numpy as np
from keras import layers
import pandas as pd

import ccxt

path15=r'D:\Python1\AI\cripto_bot\from_3_4.parquet'
path4=r"D:\Python1\AI\cripto_bot\from_2_4.parquet"
path1=r"D:\Python1\AI\cripto_bot\from_1_4.parquet"

class add_i():
    def __init__(self):
        self.columns=['timestamp','open','high','low','close','volue']

    def __call__(self,m15,fr):
                
        m15_pd=pd.DataFrame(m15,columns=self.columns)
        rsi=ta.momentum.rsi(m15_pd['close'],14)
        print(f'fr:{fr}| len:{len(m15_pd)}')
        try:
            adx=ta.trend.adx(m15_pd['high'],m15_pd['low'],m15_pd['close'],14)
        except Exception as a:
            print(a,'| ', m15_pd,"|",fr)
        bb=ta.volatility.BollingerBands(m15_pd['close'],20,2)
        bolban=bb.bollinger_pband()
        atr=ta.volatility.average_true_range(m15_pd['high'],m15_pd['low'],m15_pd['close'])
        #добавление данных
        m15_pd['rsi']=rsi
        m15_pd['adx']=adx
        m15_pd['bb']=bolban
        #m15_pd['obv']=obv
        m15_pd['atr']=atr

                
        m15_pd['timestamp']=pd.to_datetime(m15_pd['timestamp'],utc=True,unit='ms')
        start=m15_pd["timestamp"].iloc[29]

        end=m15_pd['timestamp'][len(m15_pd['timestamp'])-1]
        print(f'start:{start}')

        print(f"end:{end}")

        m15_pd=m15_pd.iloc[29:]
        #разбиение на этап

        return m15_pd

ki=ccxt.binance()
ku=ccxt.kucoin()
path15=r'D:\Python1\AI\cripto_bot\from_3_3.parquet'
path15_=r'D:\Python1\AI\cripto_bot\from_3_4.parquet'

path4=r"D:\Python1\AI\cripto_bot\from_2_3.parquet"
path4_=r"D:\Python1\AI\cripto_bot\from_2_4.parquet"

path1=r"D:\Python1\AI\cripto_bot\from_1_3.parquet"
path1_=r"D:\Python1\AI\cripto_bot\from_1_4.parquet"

path_f15=Path(r'D:\Python1\AI\cripto_bot\from_3_3.feather')
path_f4=Path(r'D:\Python1\AI\cripto_bot\from_2_3.feather')
path_f1=Path(r'D:\Python1\AI\cripto_bot\from_1_3.feather')

class parse():
    def __init__(self):
        self.data15=pd.read_parquet(path15_,engine='pyarrow')
        self.data4=pd.read_parquet(path4_,engine='pyarrow')
        self.data1=pd.read_parquet(path1_,engine='pyarrow')

        self.data15_f=pd.read_feather(path_f15)
        self.data4_f=pd.read_feather(path_f4)
        self.data1_f=pd.read_feather(path_f1)
            
        self.timeframe=['1w','4h','15m']
        self.add_indikator=add_i()
        print('11')

    def __call__(self,t):
        
        if self.data15_f.empty:
            last_=self.data15['timestamp'].iloc[-1]
            print(last_,'1')
        else:
            last_=self.data15_f['timestamp'].iloc[-1]
            print(last_,'2')

        data={}
        if t-last_>=timedelta(minutes=15):
            po=True
        #if t-datetime.datetime.fromisoformat("2021-02-11T02:45:00+00:00")>=timedelta(minutes=15):
            start=(self.data15['timestamp'].iloc[-1]+datetime.timedelta(minutes=30))
           

            #start=datetime.datetime.fromisoformat("2021-02-11T03:15:00+00:00")
            end=t+timedelta(seconds=1)
            
            with fut.ThreadPoolExecutor(3) as tr:
                            future=[tr.submit(self.daa1,fr,start,end) for fr in self.timeframe]
                            #future=[tr.submit(self.daa1,self.timeframe[2],start,end),]  
                            for k in fut.as_completed(future):
                                    fr,u=k.result()
                                    data[fr]=u
                
        else:
            start_ind=len(self.data15)
            self.data15_f.index=range(start_ind,start_ind+len(self.data15_f))
            data['15m']=pd.concat([self.data15,self.data15_f])

            start_ind=len(self.data4)
            self.data4_f.index=range(start_ind,start_ind+len(self.data4_f))
            data['4h']=pd.concat([self.data4,self.data4_f])

            start_ind=len(self.data1)
            self.data1_f.index=range(start_ind,start_ind+len(self.data1_f))
            data['1w']=pd.concat([self.data1,self.data1_f])
            po=False
        return data,po
    def parsbi(self,start_,end_,fr):
        cans=[]
        since=start_
        while since<=end_:
            
            can=ki.fetch_ohlcv("BTC/USDT",fr,since,limit=1000)
            if not can:
                break
            c1=[c for c in can if c[0]<end_]
            if not c1:
                break
            cans.extend(c1)
            since = cans[-1][0] + 1  # Передвигаем `since` на последнюю свечу
            time.sleep(0.1) 
        return cans
    
    def daa1(self,fr,start,end):
        if fr=='15m':
             start1=start - timedelta(minutes=15*100)
             fun=self.test_data15
             start=start-timedelta(minutes=15*30)
             save=self.save15
        elif fr=='4h':
             start1=start-timedelta(hours=4*100)
             fun=self.test_data4
             start=start-timedelta(hours=4*30)
             save=self.save4
             
        elif fr=='1w':
             start1=start - timedelta(days=7*100)
             fun=self.test_data1
             start=start-timedelta(days=7*30)
             save=self.save1

        else:
             raise Exception('dont correct ft in daa1')
        print(f'fr:{fr}: start:{start1}| end:{end}')
        start_ku=ki.parse8601(start1.isoformat())
        end_ku=ki.parse8601(end.isoformat())
        u=self.parsbi(start_ku,end_ku,fr)
        u=fun(fr,u,start,end)
        u=self.add_indikator(u,fr)
        u=save(u)
        #for i in u:
             #print(f'{fr}:{datetime.datetime.fromtimestamp(i[0]/1000,timezone.utc)}')
        return fr,u
    
    def save15(self,data):
        self.data15_f=pd.concat([self.data15_f,data],ignore_index=True)
        self.data15_f=self.data15_f.drop_duplicates(subset='timestamp',keep='last')
        start_ind=len(self.data15)
        self.data15_f.index=range(start_ind,start_ind+len(self.data15_f))
        data_=pd.concat([self.data15,self.data15_f]).drop_duplicates(subset='timestamp',keep='last')

        if len(self.data15_f)>=16:
            self.data15=data_
            self.data15.to_parquet(path15_)
            self.data15_f=pd.DataFrame([])
        self.data15_f=self.data15_f
        self.data15_f.to_feather(path_f15)
        return data_
    
    def save4(self,data):
        self.data4_f=pd.concat([self.data4_f,data],ignore_index=True)
        self.data4_f=self.data4_f.drop_duplicates(subset='timestamp',keep='last')
        start_ind=len(self.data4)
        self.data4_f.index=range(start_ind,start_ind+len(self.data4_f))
        data_=pd.concat([self.data4,self.data4_f]).drop_duplicates(subset='timestamp',keep='last')
        
        if len(self.data4_f)>=16:
            self.data4=data_
            
            self.data4.to_parquet(path4_)
            self.data4_f=pd.DataFrame([])
        self.data4_f.to_feather(path_f4)
        return data_
        

    def save1(self,data):
        self.data1_f=pd.concat([self.data1_f,data],ignore_index=True)
        self.data1_f=self.data1_f.drop_duplicates(subset='timestamp',keep='last')
        start_ind=len(self.data1)
        self.data1_f.index=range(start_ind,start_ind+len(self.data1_f))
        data_=pd.concat([self.data1,self.data1_f]).drop_duplicates(subset='timestamp',keep='last')
        if len(self.data1_f)>=10:
            self.data1=data_
            self.data1.to_parquet(path1_)
            self.data1_f=pd.DataFrame([])
        self.data1_f.to_feather(path_f1)
        return data_



    def parsku(self,start_,end_,timeframe):
        cans=[]
        since=start_
        while since<=end_:
            can=ku.fetch_ohlcv("BTC/USDT",timeframe,since,limit=1000)
            if not can:
                break
            c1=[c for c in can if c[0]<end_]
            if not c1:
                break
            cans.extend(c1)
            since = cans[-1][0] + 1  # Передвигаем `since` на последнюю свечу 
        return cans

    def test_data15(self,timeframe,data,start,end):
        m15=data
        dot=[]
        last=datetime.datetime.fromtimestamp(m15[0][0]/1000,datetime.timezone.utc)
        print(f'last:{last}-end:{datetime.datetime.fromtimestamp(m15[-1][0]/1000,datetime.timezone.utc)}')
        for ind,i in enumerate(m15):
        #if 10<ind<20:
         #    print(datetime.datetime.fromtimestamp(m15[ind][0]/1000,timezone.utc))
            if datetime.datetime.fromtimestamp(i[0]/1000,timezone.utc) < start:
                last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
                continue
            if ind ==0:
                continue
            if datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(minutes=15)!=last:
                print('--------------------------')
                start=last
                print(f'start1:{start.isoformat()}',end='|')
                end=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(seconds=1)
                print(f'end1:{end.isoformat()}',end='|')
                print(f"i:{datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)}| last:{last}")

                start_ku_v=ku.parse8601((start-datetime.timedelta(minutes=15*99)).isoformat())
                end_ku_v=ku.parse8601(end.isoformat())
                #print('start ku:',(last-datetime.timedelta(hours=4*175)))
                #print('end ku:',last)
                #print(f'time:{datetime.datetime.fromtimestamp(m15[ind][0]/1000,timezone.utc)} | ind:{ind} | last time:{datetime.datetime.fromtimestamp(m15[ind-1][0]/1000,timezone.utc)} | time:{last}')
                #print('start bi:',datetime.datetime.fromtimestamp( dat[-175][0]/1000,timezone.utc))
                #print('end bi:',datetime.datetime.fromtimestamp(dat[-1][0]/1000,timezone.utc))
                ku_can_v=self.parsku(start_ku_v,end_ku_v,timeframe)

                start1=datetime.datetime.fromtimestamp(ku_can_v[0][0]/1000,timezone.utc)
                
                for o in ku_can_v:
                    if datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)!=start1:
                        print(f"ERROR({datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)} : {start1})")
                        start1=datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)
                    start1+=datetime.timedelta(minutes=15)
                mean_v1=np.array(data[:100])
                pl=datetime.datetime.fromtimestamp(mean_v1[-1][0]/1000,timezone.utc)
                for o in range(1,len(mean_v1)):
                    o+=1
                    o*=-1
                    if pl-datetime.timedelta(minutes=15)!=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc):
                        print("AAAA:",o,'|',datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc))

                    pl=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc)
                mean_v1=mean_v1[:,-1]
                mean_v2=np.array(ku_can_v)[:100,-1]
                
                
                #mean_v2=mean_v2[mean_v2!=0]
                #print('v1 start:',datetime.datetime.fromtimestamp(mean_v1[0][0]/1000,timezone.utc),' | end:',datetime.datetime.fromtimestamp(mean_v1[-1][0]/1000,timezone.utc),'| len:',len(mean_v1))
                #print('v2 start:',datetime.datetime.fromtimestamp(mean_v2[0][0]/1000,timezone.utc),' | end:',datetime.datetime.fromtimestamp(mean_v2[-1][0]/1000,timezone.utc),'| len:',len(mean_v2))
                
                k=mean_v1/mean_v2
                #print("v1:",mean_v1[:5],"| v2:",mean_v2[:5])
                k=k[k!=np.inf].mean()
                #print('k not normaliz:',k,end='|')
                k=k
                
                #print('k:',k)
                #print('norm:',(mean_v2*k)[:10])
                
                # start_ku=ku.parse8601(start.isoformat())
                # end_ku=ku.parse8601(end.isoformat())

                # ku_can=self.parsku(start_ku,end_ku,timeframe)
                ku_can=ku_can_v[100:]
                #print(f'len:{len(ku_can)}')
                #try:
                    #print(f'strt:{datetime.datetime.fromtimestamp(ku_can[0][0]/1000,datetime.timezone.utc)}')
                    #print(f'end:{datetime.datetime.fromtimestamp(ku_can[-1][0]/1000,datetime.timezone.utc)}')
                #except IndexError:
                    #print(int((end-start).total_seconds()))
                    #'''for kline in range(int((end-start).total_seconds()/)):
                     #   print(kline)'''
                #timestamps = [c[0] for c in m15]  # Получаем список всех timestamp
                ku_can_new=[]
                for inde,o in enumerate(ku_can):
                    if inde==0:
                        o[1]=data[-1][4]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    if inde+1==len(ku_can):
                        o[4]=i[1]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    o[5]=o[5]*k
                    ku_can_new.append(o)
                
                #print('last close:',m15[-1][4],' new open:',ku_can_new[0][1],'| new close:',ku_can_new[-1][4],' last open:',i[1])
                ku_can=ku_can_new
                dot.extend(ku_can)
                dot.append(i)

            #da['15m']['15m']
            #for candle in ku_can:
            #    if candle[0] not in timestamps:
            #        index = bisect.bisect_right(timestamps, candle[0])  # Найти правильное место
            #        m15.insert(index, candle)
            
            
                '''
                
                ge=requests.get('https://api.kucoin.com/api/v1/market/candles',params)
                ge.raise_for_status()
                print(f'strt:{datetime.datetime.fromtimestamp(ge[0][0])}')
                print(f'end:{datetime.datetime.fromtimestamp(ge[-1][6])}')
                #?symbol=BTC-USDT&type=1hour&startAt=1679616000&endAt=1679702400
                
                '''

            else:
                dot.append(i)
            last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
        return dot


    def test_data4(self,timeframe,data,start,end):
        m15=data
        dot=[]
        last=datetime.datetime.fromtimestamp(m15[0][0]/1000,datetime.timezone.utc)
        for ind,i in enumerate(m15):

            if datetime.datetime.fromtimestamp(i[0]/1000,timezone.utc) < start:
                last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
                continue
            if ind ==0:
                continue
            if datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(hours=4)!=last:
                print('--------------------------')
                start=last
                print(f'start1:{start.isoformat()}',end='|')
                end=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(seconds=1)
                print(f'end1:{end.isoformat()}',end='|')
                print(f"i:{datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)}| last:{last}")

                start_ku_v=ku.parse8601((start-datetime.timedelta(hours=4*99)).isoformat())
                end_ku_v=ku.parse8601(end.isoformat())

                ku_can_v=self.parsku(start_ku_v,end_ku_v,timeframe)

                start1=datetime.datetime.fromtimestamp(ku_can_v[0][0]/1000,timezone.utc)

                for o in ku_can_v:
                    if datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)!=start1:
                        print(f"ERROR({datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)} : {start1})")
                        start1=datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)
                    start1+=datetime.timedelta(hours=4)
                mean_v1=np.array(data[:100])
                pl=datetime.datetime.fromtimestamp(mean_v1[-1][0]/1000,timezone.utc)
                for o in range(1,len(mean_v1)):
                    o+=1
                    o*=-1
                    if pl-datetime.timedelta(hours=4)!=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc):
                        print("AAAA:",o,'|',datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc))

                    pl=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc)
                mean_v1=mean_v1[:,-1]
                mean_v2=np.array(ku_can_v)[:100,-1]

                k=mean_v1/mean_v2
                k=k[k!=np.inf].mean()
                ku_can=ku_can_v[100:]
                ku_can_new=[]
                for inde,o in enumerate(ku_can):
                    if inde==0:
                        o[1]=data[-1][4]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    if inde+1==len(ku_can):
                        o[4]=i[1]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    o[5]=o[5]*k
                    ku_can_new.append(o)

                ku_can=ku_can_new
                dot.extend(ku_can)
                dot.append(i)

            else:
                dot.append(i)
            last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
        return dot
    
    def test_data1(self,timeframe,data,start,end):
        m15=data
        dot=[]
        last=datetime.datetime.fromtimestamp(m15[0][0]/1000,datetime.timezone.utc)
        for ind,i in enumerate(m15):

            if datetime.datetime.fromtimestamp(i[0]/1000,timezone.utc) < start:
                last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
                continue
            if ind ==0:
                continue
            if datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(days=7)!=last:
                print('--------------------------')
                start=last
                print(f'start1:{start.isoformat()}',end='|')
                end=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)-datetime.timedelta(seconds=1)
                print(f'end1:{end.isoformat()}',end='|')
                print(f"i:{datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)}| last:{last}")

                start_ku_v=ku.parse8601((start-datetime.timedelta(days=7*99)).isoformat())
                end_ku_v=ku.parse8601(end.isoformat())

                ku_can_v=self.parsku(start_ku_v,end_ku_v,timeframe)

                start1=datetime.datetime.fromtimestamp(ku_can_v[0][0]/1000,timezone.utc)

                for o in ku_can_v:
                    if datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)!=start1:
                        print(f"ERROR({datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)} : {start1})")
                        start1=datetime.datetime.fromtimestamp(o[0]/1000,timezone.utc)
                    start1+=datetime.timedelta(days=7)
                mean_v1=np.array(data[:100])
                pl=datetime.datetime.fromtimestamp(mean_v1[-1][0]/1000,timezone.utc)
                for o in range(1,len(mean_v1)):
                    o+=1
                    o*=-1
                    if pl-datetime.timedelta(days=7)!=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc):
                        print("AAAA:",o,'|',datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc))

                    pl=datetime.datetime.fromtimestamp(mean_v1[o][0]/1000,timezone.utc)
                mean_v1=mean_v1[:,-1]
                mean_v2=np.array(ku_can_v)[:100,-1]

                k=mean_v1/mean_v2
                k=k[k!=np.inf].mean()
                ku_can=ku_can_v[100:]
                ku_can_new=[]
                for inde,o in enumerate(ku_can):
                    if inde==0:
                        o[1]=data[-1][4]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    if inde+1==len(ku_can):
                        o[4]=i[1]
                        o[2]=max(o[2],max(o[4],o[1]))
                        o[3]=min(o[3],min(o[4],o[1]))
                    o[5]=o[5]*k
                    ku_can_new.append(o)

                ku_can=ku_can_new
                dot.extend(ku_can)
                dot.append(i)

            else:
                dot.append(i)
            last=datetime.datetime.fromtimestamp(i[0]/1000,datetime.timezone.utc)
        return dot

 