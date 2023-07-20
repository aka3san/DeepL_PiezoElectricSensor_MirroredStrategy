import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from scipy.optimize import curve_fit
import random

#ガウスフィッティングで使う関数
def gauss2(x, *params):
    num_func = len(params)//3
    
    y_sum = 0
    for i in range(num_func):
        y = params[i] * np.exp(-(x-params[i+num_func*1])**2/2*(params[i+num_func*2]**2))
        y_sum += y
    return y_sum

#一周期のデータを格納するクラス
class OneCycleData:
    _data: pd.DataFrame = None #一周期のデータフレーム
    _cycle: float = None #一周期の時間
    _maxValue: float = None #一周期の最大値
    _minValue: float = None #一周期の最小値
    _area: float = None #面積
    
    #コンストラクタ
    def __init__(self, data_oneCycle: pd.DataFrame):
        dataList = data_oneCycle.copy().to_numpy().tolist()
        self._data = pd.DataFrame(dataList)
        self.convertDataIntoPositive() #データの最小値を0にする
        self.converDataInto0Start() #xの基準を0にする
        self._maxValue = self._data.max(axis=0)[1]
        self._maxValue_x = 0
        for i in range(len(self._data.iloc[:,1])):
            if self._data.iloc[i,1] == self._maxValue:
                self._maxValue_x = self._data.iloc[i,0] 
        self._minValue = self._data.min(axis=0)[1]
        self._cycle = self._data.iloc[len(self._data.index)-1,0] 
        self._area = self.calculate_area(self._data)
    
    def get_data(self):
        return self._data.copy()
    def get_maxValue(self):
        return self._maxValue
    def get_minValue(self):
        return self._minValue
    def get_cycle(self):
        return self._cycle
    def get_area(self):
        return self._area
    def get_peak(self, plot=False):
        #ガウスフィッティングを行いピーク値検出
        maxIndex = len(self._data.index)//3
        guess = [1,1,1,1,self._maxValue_x,0,0,0,2,2,2,2]
        peakList = [-1,-1,-1,-1]
        #ピーク値がデータの範囲内に収まるまで繰り返す
        popt = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        while self.check_withinData(peakList) == False:
            try:
                popt, pcov = curve_fit(gauss2, self._data.iloc[:,0], self._data.iloc[:,1], guess)
            except:
                pass
            A, B, C, D,mu_x1, mu_x2, mu_x3,mu_x4, sigma_x1, sigma_x2, sigma_x3,sigma_x4= popt
            peakList = [mu_x1, mu_x2, mu_x3, mu_x4] #極大値のｘ座標のリスト
            #初期値の更新
            guess = [random.random(), random.random(), random.random(), random.random(), self._maxValue_x, random.uniform(0.05,0.8), random.uniform(0.05,0.8), random.uniform(0.05,0.8), 2, 2, 2, 2]
        peakList.sort()
        
         #plot=Trueが指定されたとき、導出したガウス関数をプロットする
        if plot:
            def gauss_optx1(x):
                z = A * np.exp(-(x-mu_x1)**2/2*(sigma_x1**2))
                return z
            y1 = [gauss_optx1(x) for x in self._data.iloc[:, 0]]
            def gauss_optx2(x):
                z = B * np.exp(-(x-mu_x2)**2/2*(sigma_x2**2))
                return z
            y2 = [gauss_optx2(x) for x in self._data.iloc[:, 0]]
            def gauss_optx3(x):
                z = C * np.exp(-(x-mu_x3)**2/2*(sigma_x3**2))
                return z
            y3 = [gauss_optx3(x) for x in self._data.iloc[:, 0]]
            def gauss_optx4(x):
                z = D * np.exp(-(x-mu_x4)**2/2*(sigma_x4**2))
                return z
            y4 = [gauss_optx4(x) for x in self._data.iloc[:, 0]]
            graph = plt.figure().add_subplot(1,1,1)
            graph.plot(self._data.iloc[:,0], y1)
            graph.plot(self._data.iloc[:,0], y2)
            graph.plot(self._data.iloc[:,0], y3)
            graph.plot(self._data.iloc[:,0], y4)
            graph.plot(self._data.iloc[:,0], self._data.iloc[:,1])
            
        #極小値も計算する
        i = 0
        mu_x1_idx = int(peakList[0]//0.003)
        while self._data.iloc[mu_x1_idx +i,1]  < self._data.iloc[mu_x1_idx +i+1,1] or self._data.iloc[mu_x1_idx +i+1,1]  > self._data.iloc[mu_x1_idx +i+2,1]:
            #極小値が見つからない場合は極大値の中間地点とする
            if(self._data.iloc[mu_x1_idx +i+1,0] >= peakList[1]):
                localMinimum = (peakList[0]+peakList[1])/2
                i = int(localMinimum//0.003) - 1
                break
            i += 1
        peakList.append(self._data.iloc[mu_x1_idx+i+1,0])
        i=0
        mu_x2_idx = int(peakList[1]//0.003)
        while self._data.iloc[mu_x2_idx+i,1]  < self._data.iloc[mu_x2_idx+i+1,1] or self._data.iloc[mu_x2_idx+i+1,1]  > self._data.iloc[mu_x2_idx+i+2,1]:
            if(self._data.iloc[mu_x2_idx+i+1,0] >= peakList[3]):
                localMinimum = (peakList[0]+peakList[1])/2
                i = int(localMinimum//0.003) - 1
                break
            i += 1
        peakList.append(self._data.iloc[mu_x2_idx+i+1,0])
        peakList.sort()
        peakList.pop()
        if len(peakList) != 5:
            raise Exception("ピークの数に異常があります。")
        return peakList
    
    #xの値が一周期のデータの範囲内か判定する。
    def check_withinData(self, xList):
        for x in xList:
            if x <= 0 or x >= self._cycle:
                return False
        else:
            True
        
    #一周期の全体の面積を計算して返す。
    def calculate_area(self,data_oneCycle: pd.DataFrame):
        deltaTime = 0.003
        area = 0
        for y in list(data_oneCycle.iloc[:, 1]):
            area += y*deltaTime
        return area
    
    #元データを最低電圧を0としたデータに変換
    def convertDataIntoPositive(self):
        minValue = self._data.min(axis=0)[1]
        self._data.iloc[:,1] = [y - minValue for y in self._data.iloc[:,1]]
    #横軸を0からにする
    def converDataInto0Start(self):
        initialValue = self._data.iloc[0,0]
        self._data.iloc[:,0] = [x - initialValue for x in self._data.iloc[:,0]]
        


#電圧データ全体から単一波を返す関数
def get_oneWave(originalData: pd.DataFrame, allWaves_plot=False, oneWave_plot=False, fileName="onWave_plot"):
    data = originalData.copy()
    data.iloc[:,1] = denoise(data.iloc[:,1], wavelet="db1", level=4)
    originalData.iloc[:,1] = denoise(originalData.iloc[:,1], wavelet="db36", level=3)
    cycleList = []
    #１周期の開始場所を見つける
    i=0
    while i < len(data.index)-2:
        #極大値の時
        if data.iloc[i+1,1] > 0 and (data.iloc[i+1,1] - data.iloc[i,1] > 0 and data.iloc[i+2,1] - data.iloc[i+1,1] == 0):
            j = i               
            #グラフを逆にたどって初めて極小値が現れる場所
            while (originalData.iloc[j-1,1] - originalData.iloc[j,1] <= 0 or originalData.iloc[j+1,1] - originalData.iloc[j,1] <= 0)  and j >= 1:
                j = j-1
            if(j > 0):
                cycleList.append(j)
            while data.iloc[i+1,1] > 0 and i < len(data.index)-2:
                i = i+1
        else:
            i = i+1
    #一周期ずつデータを格納していく
    oneCycleDatas = []
    meanValue = 0
    for i in range(len(cycleList)-1):
        oneCycleDatas.append(OneCycleData(originalData[cycleList[i]:cycleList[i+1]]))
        meanValue = meanValue + oneCycleDatas[i].get_area()
    meanValue = meanValue/(len(cycleList)-1)
    #最も平均に近い面積の単一波を決定する。
    oneCycleDataCloseToAverage = oneCycleDatas[0]
    i = 0
    i_last = 0
    for oneCycleData in oneCycleDatas:
        if abs(oneCycleData.get_area() - meanValue) < abs(oneCycleDataCloseToAverage.get_area() - meanValue):
            oneCycleDataCloseToAverage = oneCycleData
            i_last = i
        i += 1
    #単一波一覧をプロット
    if allWaves_plot:
        figure_original = plt.figure().add_subplot(1,1,1)
        figure_original.set_title("Original Data")
        figure_original.plot(originalData.iloc[:,0], originalData.iloc[:,1])
        i=0
        figure_cut = plt.figure()
        figure_cut.suptitle("Divided Original Data")
        for i in range(len(cycleList)-1):
            figure_cut.add_subplot(1,len(cycleList)-1,i+1).plot(originalData.iloc[cycleList[i]:cycleList[i+1],1])
        
    if oneWave_plot:
        #figure_db1 = plt.figure().add_subplot(1,1,1)
        #figure_db1.set_title("Noise-processed Data In DB1")
        #figure_db1.plot(data.iloc[:,0], data.iloc[:,1])
        #figure_db1.savefig(fileName+"_db")
        figure_one = plt.figure()
        figure_one_ax = figure_one.add_subplot(1,1,1)
        figure_one_ax.set_title("Single Extracted From Original Data")
        figure_one_ax.plot(oneCycleDataCloseToAverage.get_data().iloc[:,0], oneCycleDataCloseToAverage.get_data().iloc[:,1])
        figure_one.savefig(f"figures/{fileName}")
        plt.close()
        #plt.close()
    #最適な単一波を返す
    return oneCycleDataCloseToAverage

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='sym20',level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

#テスト
#csv = pd.read_csv("/Users/inayoshikouya/Downloads/DeepL_PiezoElectricSensor/BPDatas/KI_20230523_1551.CSV")
#oneWave = get_oneWave(csv,allWaves_plot=True, oneWave_plot=True)



