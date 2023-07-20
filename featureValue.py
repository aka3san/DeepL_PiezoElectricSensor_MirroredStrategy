import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import preprocess as pp
import math
import subprocess
import nitime.algorithms as tsa
from scipy import integrate
from scipy import interpolate



#CSVを受け取って、特徴量クラスを返す。
def convertToFV(originalData: pd.DataFrame, plot=False, fileName="onWave_plot"):
    oneCycleData = pp.get_oneWave(originalData,oneWave_plot=plot, allWaves_plot=False, fileName=fileName)   
    featureValues = FeatureValues(oneCycleData)
    return featureValues

#特徴量をまとめたクラス
class FeatureValues:
    def __init__(self, oneCycleData: pp.OneCycleData):        
        #各特徴量
        self.data = oneCycleData.get_data()
        self.data.iloc[:,0] = [round(x,3) for x in self.data.iloc[:,0]]
        self.AG: float = None #
        self.AE_AG: float = None #
        self.EG_AG: float = None #
        self.AC_AG: float = None #
        self.CE_AG: float = None #
        self.AB_AC: float = None #
        self.BC_AC: float = None #
        self.CD_CE: float = None #
        self.DE_CE: float = None #
        self.EF_EG: float = None #
        self.FG_EG: float = None #
        self.H: float = None #
        self.f_H: float = None #
        self.g_H: float = None #
        self.i_H: float = None #
        self.H_AB: float = None #
        self.S: float = None 
        self.S_sys: float = None #
        self.S_dia: float = None #
        
        
     #時間に関する特徴量の計算
        #AGの計算
        self.AG = oneCycleData.get_cycle()
        #AE_AGの計算
        peakList = oneCycleData.get_peak(plot=True)
        self.AE_AG = peakList[3]/self.AG
        #EG_AGの計算
        EG = self.AG - peakList[3]
        self.EG_AG = EG/self.AG
        #AC_AGの計算
        self.AC_AG = peakList[1]/self.AG
        #CE_AGの計算
        CE = peakList[3] - peakList[1]
        self.CE_AG = CE/self.AG
        #AB_ACの計算
        self.AB_AC = peakList[0]/peakList[1]
        #BC_ACの計算
        BC = peakList[1] - peakList[0]
        self.BC_AC = BC/peakList[1]
        #CD_CEの計算
        CD = peakList[2] - peakList[1]
        self.CD_CE = CD/CE
        #DE_CEの計算
        DE = peakList[3] - peakList[2]
        self.DE_CE = DE/CE
        #EF_EGの計算
        EG = self.AG - peakList[3]
        EF = peakList[4] -peakList[3]
        self.EF_EG = EF/EG
        #FG_EGの計算
        FG = self.AG - peakList[4]
        self.FG_EG = FG/EG
     #時間に関する特徴量の計算
        #Hの計算        
        B_idx = self.XvalueToIndex(peakList[0])
        self.H = self.data.iloc[B_idx,1]
        #f_Hの計算
        f_idx = self.XvalueToIndex(peakList[4])
        self.f_H = self.data.iloc[f_idx,1]/self.H
        #g_Hの計算
        g_idx = self.XvalueToIndex(peakList[3])
        self.g_H = self.data.iloc[g_idx,1]/self.H
        #i_Hの計算
        i_idx = self.XvalueToIndex(peakList[1])
        self.i_H = self.data.iloc[i_idx,1]/self.H
        #H_ABの計算
        self.H_AB = self.H/peakList[0]
     
     #面積に関する特徴量の計算
        #Sの計算
        self.S = oneCycleData.get_area()
        area_sys = 0
        area_dia = 0
        deltaTime = self.data.iloc[1,0] - self.data.iloc[0,0]
        for i in range(0, self.XvalueToIndex(peakList[3])):
            area_sys += self.data.iloc[i,1]*deltaTime
        for i in range(self.XvalueToIndex(peakList[3]), len(self.data)):
            area_dia += self.data.iloc[i,1]*deltaTime
        self.S_sys = area_sys/self.S
        self.S_dia = area_dia/self.S
      
      #周波数領域の特徴量の計算
        dt = 0.003
        fs = 1/dt
        y = self.data.copy()
        y = y.to_numpy()[:,1]
        freq, P_mt, nu = tsa.multi_taper_psd(y, Fs=fs)  # MTM
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("PSD [W/Hz]")
        ax.plot(freq, P_mt)
        f = interpolate.interp1d(freq, P_mt, kind='linear')
        x_list = [x*0.5 for x in range(0,21) ]
        S_list = []
        for i in range(0, len(x_list)-1):
            integ = integrate.quad(f, x_list[i], x_list[i+1])
            S_list.append(integ[0])
        for i in range(0,len(S_list)):
            exec(f"self.PSD{i+1}={S_list[i]}")
        
    #xの値から一周期データのインデックスを返す。
    def XvalueToIndex(self, x):
        dataList = self.data.iloc[:,0]
        for i in range(0, len(dataList)-1):
            if dataList[i] <= x and x < dataList[i+1]:
                return i
        else:
            print("xの値がデータの範囲外です。")
            #raise Exception("xの値がデータの範囲外です。")
    def printFVs(self):
        return

#テスト
#csv = pd.read_csv("C:/Users/azlab/OneDrive - 国立大学法人東海国立大学機構/ドキュメント/PZT圧電センサ/BPDatas/KI_20230222_1810.CSV")
#oneCycleData = pp.get_oneWave(csv,oneWave_plot=False, allWaves_plot=False)   
#fv = FeatureValues(oneCycleData)
#print(vars(fv))


