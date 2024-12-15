# from stock_data_tools import last_train_data,deleteNullFile,delet,init,compare_date
import os
import pandas as pd
from multiprocessing import Pool,Lock
# from utils.path_and_canshu import basic_data_save_path,other_file_path,参数化数据,周期转化后
import talib  
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
def 参数生成_all(path,goal_path,multiprocess=True):
    '''
    其他参数
    :path:数据源的地址
    :goal_path:目标储存位置
    '''
    codelist = os.listdir(path)
    for i in range(len(codelist)):
        if codelist[i] not in os.listdir(goal_path):
            codelist[i] = path + codelist[i] + ',' + goal_path

    if multiprocess:
        lock = Lock()
        with Pool(10, initializer=tqdm.set_lock, initargs=(lock,)) as pool:
            with tqdm(total=len(codelist), desc="Processing") as pbar:
                for _ in pool.imap_unordered(参数生成, codelist):
                    pbar.update(1)
    else:
        for i in tqdm(codelist, desc="Processing"):
            参数生成(i)
    # 参数生成(codelist[0])
    print('参数化完成')
def 参数生成(filename):
    goal_path=filename.split(',')[1]
    filename=filename.split(',')[0]
    df=pd.read_parquet(filename)
    df['CDL2CROWS'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDL3INSIDE'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
    df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
    df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
    df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
    df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
    df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
    df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
    df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
    df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
    df['CDLHARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])
    df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])
    df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDLINNECK'] = talib.CDLINNECK(df['open'], df['high'], df['low'], df['close'])
    df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['CDLKICKING'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
    df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])
    df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])
    df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLLONGLINE'] = talib.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])
    df['CDLMATHOLD'] = talib.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['CDLONNECK'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])
    df['CDLPIERCING'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
    df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])
    df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
    df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])
    df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
    df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])
    df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])
    df['CDLTAKURI'] = talib.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])
    df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])
    df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])
    df['CDLTRISTAR'] = talib.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])
    df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])

    df['DEMA'] = talib.DEMA(df['close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['close'], timeperiod=30)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])
    df['KAMA'] = talib.KAMA(df['close'], timeperiod=30)
    df['MA'] = talib.MA(df['close'], timeperiod=30, matype=0)
    # df['MAVP'] = talib.MAVP(df['close'])
    df['MIDPOINT'] = talib.MIDPOINT(df['close'], timeperiod=14)
    df['MIDPRICE'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=14)
    df['SAR'] = talib.SAR(df['high'], df['low'])
    df['SAREXT'] = talib.SAREXT(df['high'], df['low'])
    df['SMA'] = talib.SMA(df['close'], timeperiod=30)
    df['T3'] = talib.T3(df['close'], timeperiod=5, vfactor=0)
    df['TEMA'] = talib.TEMA(df['close'], timeperiod=30)
    df['TRIMA'] = talib.TRIMA(df['close'], timeperiod=30)
    df['WMA'] = talib.WMA(df['close'], timeperiod=30)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=14)
    df['APO'] = talib.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    df['AROONOSC'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
    df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['CMO'] = talib.CMO(df['close'], timeperiod=14)
    df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)
    df['MACDmacdhist'],df['MACDmacdsignal'],df['MACDmacd'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACDEXTmacdhist'],df['MACDEXTmacdsignal'],df['MACDEXTmacd'] = talib.MACDEXT(df['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df['MACDFIXmacdhist'],df['MACDFIXmacdsignal'],df['MACDFIXmacd'] = talib.MACDFIX(df['close'], signalperiod=9)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=14)
    df['PPO'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)
    df['ROCP'] = talib.ROCP(df['close'], timeperiod=10)
    df['ROCR'] = talib.ROCR(df['close'], timeperiod=10)
    df['ROCR100'] = talib.ROCR100(df['close'], timeperiod=10)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
    df['AVGPRICE'] = talib.AVGPRICE(df['open'], df['high'], df['low'], df['close'])
    df['MEDPRICE'] = talib.MEDPRICE(df['high'], df['low'])
    df['TYPPRICE'] = talib.TYPPRICE(df['high'], df['low'], df['close'])
    df['WCLPRICE'] = talib.WCLPRICE(df['high'], df['low'], df['close'])
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close'])
    df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close'])
    df['HT_PHASORquadrature'],df['HT_PHASORinphase'] = talib.HT_PHASOR(df['close'])
    df['HT_SINEleadsine'],df['HT_SINEsine'] = talib.HT_SINE(df['close'])
    df['STOCHslowd'],df['STOCHslowk'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCHFfastd'],df['STOCHFfastk'] = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSIfastd'],df['STOCHRSIfastk'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['AROONaroonup'],df['AROONaroondown'] = talib.AROON(df['high'], df['low'], timeperiod=14)
    # df['MAMAfama'],df['MAMAmama'] = talib.MAMA(df['close'], fastlimit=3, slowlimit=5)
    df['BBANDSlowerband'],df['BBANDSmiddleband'],df['BBANDSupperband'] = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0) #stock.drop(changeratelist,axis=1,inplace=True)
    df['BETA'] = talib.BETA(df['high'], df['low'], timeperiod=5)
    df['CORREL'] = talib.CORREL(df['high'], df['low'], timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(df['close'], timeperiod=14)
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=14)
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)
    df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=5, nbdev=1)
    df['TSF'] = talib.TSF(df['close'], timeperiod=14)
    df['VAR'] = talib.VAR(df['close'], timeperiod=5, nbdev=1)
    df.replace(float('inf'),0,inplace=True)
    df.replace(float('-inf'),0,inplace=True)
    # drop_columns=['CDLHOMINGPIGEON_-2s','VAR_-1s','VAR_-2s','LINEARREG_SLOPE_-1s','LINEARREG_SLOPE_-2s','LINEARREG_ANGLE_-1s','LINEARREG_ANGLE_-2s','BETA_-1s','BETA_-2s','STOCHRSIfastd_-1s','STOCHRSIfastd_-2s','STOCHslowk_-1s','STOCHslowk_-2s','HT_PHASORquadrature_-1s','HT_PHASORquadrature_-2s','ADOSC_-1s','ADOSC_-2s','ULTOSC_-1s','ULTOSC_-2s','PPO_-1s','PPO_-2s','MFI_-1s','MFI_-2s','MACDEXTmacdhist_-1s','MACDEXTmacdhist_-2s','MACDEXTmacdsignal_-1s','MACDEXTmacdsignal_-2s','MACDEXTmacd_-1s','MACDEXTmacd_-2s','CCI_-1s','CCI_-2s','APO_-1s','APO_-2s','CDLTRISTAR_-1s','CDLTRISTAR_-2s','CDLUNIQUE3RIVER_-1s','CDLUNIQUE3RIVER_-2s','CDLUPSIDEGAP2CROWS_-1s','CDLUPSIDEGAP2CROWS_-2s','CDLTASUKIGAP_-1s','CDLTASUKIGAP_-2s','CDLSTICKSANDWICH_-1s','CDLSTICKSANDWICH_-2s','CDLRISEFALL3METHODS_-1s','CDLRISEFALL3METHODS_-2s','CDLSEPARATINGLINES_-1s','CDLSEPARATINGLINES_-2s','CDLMORNINGSTAR_-2s','CDLONNECK_-1s','CDLONNECK_-2s','CDLMATCHINGLOW_-2s','CDLMATHOLD_-1s','CDLMATHOLD_-2s','CDLMORNINGDOJISTAR_-1s','CDLMORNINGDOJISTAR_-2s','CDLKICKING_-1s','CDLKICKING_-2s','CDLKICKINGBYLENGTH_-1s','CDLKICKINGBYLENGTH_-2s','CDLLADDERBOTTOM_-1s','CDLLADDERBOTTOM_-2s','CDLINNECK_-2s','CDLIDENTICAL3CROWS_-1s','CDLIDENTICAL3CROWS_-2s','CDLHIKKAKEMOD_-1s','CDLHIKKAKEMOD_-2s','CDLGAPSIDESIDEWHITE_-1s','CDLGAPSIDESIDEWHITE_-2s','CDLEVENINGDOJISTAR_-1s','CDLCONCEALBABYSWALL_-1s','CDLCONCEALBABYSWALL_-2s','CDLCOUNTERATTACK_-1s','CDLCOUNTERATTACK_-2s','CDLBREAKAWAY_-1s','CDLBREAKAWAY_-2s','CDL3STARSINSOUTH_-1s','CDL3STARSINSOUTH_-2s','CDL3WHITESOLDIERS_-1s','CDL3WHITESOLDIERS_-2s','CDLABANDONEDBABY_-1s','CDLABANDONEDBABY_-2s','CDL3LINESTRIKE_-1s','CDL3LINESTRIKE_-2s','CDL3BLACKCROWS_-1s','CDL3BLACKCROWS_-2s','MOM','CDLUPSIDEGAP2CROWS','CDLTRISTAR','CDLTASUKIGAP','CDLSTICKSANDWICH','CDLRISEFALL3METHODS','CDLSEPARATINGLINES','CDLONNECK','CDLMATHOLD','CDLMORNINGDOJISTAR','CDLKICKING','CDLKICKINGBYLENGTH','CDLLADDERBOTTOM','CDLIDENTICAL3CROWS','CDLHIKKAKEMOD','CDLGAPSIDESIDEWHITE','CDLEVENINGDOJISTAR','CDLCONCEALBABYSWALL','CDLCOUNTERATTACK','CDLBREAKAWAY','CDL3STARSINSOUTH','CDL3WHITESOLDIERS','CDLABANDONEDBABY','CDL3LINESTRIKE','CDL2CROWS','CDL3BLACKCROWS','CDL3INSIDE','CDLEVENINGSTAR','CDLINNECK','CDLSTALLEDPATTERN','CDLUNIQUE3RIVER','CDL3INSIDE_-1s','CDLADVANCEBLOCK_-2s','CDLDRAGONFLYDOJI_-2s','CDLEVENINGDOJISTAR_-2s','CDLEVENINGSTAR_-1s','CDLEVENINGSTAR_-2s','CDLGRAVESTONEDOJI_-1s','CDLGRAVESTONEDOJI_-2s','CDLMATCHINGLOW_-1s','CDLMORNINGSTAR_-1s','CDLPIERCING_-1s','CDLPIERCING_-2s','CDLSHOOTINGSTAR_-1s','CDLSHOOTINGSTAR_-2s','CDLSTALLEDPATTERN_-1s','CDLSTALLEDPATTERN_-2s','CDLTAKURI_-1s','STOCHslowd_-1s','STOCHslowd_-2s','STOCHFfastk_-1s','STOCHFfastk_-2s']
    drop_columns=['MOM','CDLUPSIDEGAP2CROWS','CDLTRISTAR','CDLTASUKIGAP','CDLSTICKSANDWICH','CDLRISEFALL3METHODS','CDLSEPARATINGLINES','CDLONNECK','CDLMATHOLD','CDLMORNINGDOJISTAR','CDLKICKING','CDLKICKINGBYLENGTH','CDLLADDERBOTTOM','CDLIDENTICAL3CROWS','CDLHIKKAKEMOD','CDLGAPSIDESIDEWHITE','CDLEVENINGDOJISTAR','CDLCONCEALBABYSWALL','CDLCOUNTERATTACK','CDLBREAKAWAY','CDL3STARSINSOUTH','CDL3WHITESOLDIERS','CDLABANDONEDBABY','CDL3LINESTRIKE','CDL2CROWS','CDL3BLACKCROWS','CDL3INSIDE','CDLEVENINGSTAR','CDLINNECK','CDLSTALLEDPATTERN','CDLUNIQUE3RIVER']
    # df.to_csv(goal_path+filename.split('\\')[-1],encoding='gb18030',index=None,float_format='%.4f')
    # for i in range(len(averagelist)):
    #     stock[str(list(stock.columns)[averagelist[i]])] =stock[str(list(stock.columns)[averagelist[i]])]/stock['close']  #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
    # for i in changeratelist:
    #     k=stock[i].shift(1)
    #     k.drop([stock.index[0]],inplace=True)
    #     stock.drop([stock.index[0]],inplace=True)
        # stock[i+'_rate'] =(stock[i]-k)/k #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
    # stock.drop(changeratelist,axis=1,inplace=True)
    # stock.replace(float('inf'),0,inplace=True)
    # stock.replace(float('-inf'),0,inplace=True)
    # for i in range(9,len(df.columns)):
    #     df.iloc[:,i:i+1]=df.iloc[:,i:i+1]/(np.sum(df.iloc[:,i:i+1])/len(df))
        # df.iloc[:,i:i+1]=df.iloc[:,i:i+1]/df['close']
        # df[list(df.columns)[i]+'_-1s']=(df.iloc[:,i:i+1]-df.iloc[:,i:i+1].shift(1))/df.iloc[:,i:i+1].shift(1)
        # stock[list(stock.columns)[i]+'_-1s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-1']
        # df[list(df.columns)[i]+'_-2s']=(df.iloc[:,i:i+1]-df.iloc[:,i:i+1].shift(2))/df.iloc[:,i:i+1].shift(2)
        # stock[list(stock.columns)[i]+'_-2s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-2']
    # df['复权涨跌幅']=(df['close']-df['close'].shift(1))/df['close'].shift(1)
    df.drop(drop_columns,axis=1,inplace=True)
    df.replace(float('inf'),0,inplace=True)
    df.replace(float('-inf'),0,inplace=True)
    df=df.fillna(0)
    df.replace('None',0,inplace=True)
    # rename_dict={'open':'open_后复权','clsoe':'close_后复权','high':'high_后复权','low':'low_后复权','volume':'volume（手）','amount':'成交额（千元）','涨跌幅':'涨跌额','涨跌额':'涨跌幅'}
    # df.rename(columns=rename_dict,inplace=True)
    df=df[100:]
    # df.dropna(inplace=True)
    # lock.acquire()
    if len(df)>400:
        df.to_parquet(goal_path+filename.split('/')[-1])
    # lock.release()
    return 0
if __name__ == '__main__':
    path='./factor_data/'
    for filename in os.listdir(path):
        os.remove(path+filename)
    参数生成_all('./basic_data/',path,False)