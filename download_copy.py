import json
from urllib.request import urlopen
import requests
import os
import pandas as pd
import time
import akshare as ak
from multiprocessing import Pool,Lock
from tqdm import tqdm
other_file_path='./'
proxies = { "http": None, "https": None}
s=requests.session()
s.trust_env=False
s.proxies=proxies
def delet(roottdir):
    '''
    删除路径中所有文件
    :filename:文件路径
    '''
    filelist=os.listdir(roottdir)                #列出该目录下的所有文件名
    for f in filelist:
        filepath = os.path.join(roottdir, f )   #将文件名映射成绝对路劲
        if os.path.isfile(filepath):            #判断该文件是否为文件或者文件夹
            os.remove(filepath)                 #若为文件，则直接删除
def deleteNullFile(path):
    '''删除所有大小为0的文件'''
    files = os.listdir(path)
    for file in files:
        if os.path.getsize(path+file)  < 2000:   #获取文件大小
            os.remove(path+file)
            print(file + " deleted.")
    print('deleteNullFile complete')
def init(l):
	global lock
	lock = l
def download_basic_data(inputfile):
    # try:
    basic_data_save_path=inputfile.split(',')[1]
    inputfile=inputfile.split(',')[0]
    inputfile=str(inputfile)
    if len(inputfile)<6:
        inputfile='0'*(6-len(inputfile))+inputfile
    if inputfile[0]=='6':
        down_num='1.'+inputfile
        inputfile='SH'+inputfile+'.parquet'
    else:
        down_num='0.'+inputfile
        inputfile='SZ'+inputfile+'.parquet'
    if inputfile not in os.listdir(basic_data_save_path):
        # url='http://80.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101'
        # url='http://68.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&end=20500101&lmt=1000000'
        url ='http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid='+down_num+'&beg=0&end=20500000'
        # url ='http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid=1.600000&beg=0&end=20500000&_=1606616431926'
        # content=urlopen(url).read()
        content=s.get(url).content
        if len(content)==0:
           content=s.get(url).content
        if len(content)==0:
            content=s.get(url).content
        if len(content)==0:
            print('HTTP,Error '+inputfile)
            return -1
        content=content.decode('utf-8','ignore').replace('\n','')
        content=json.loads(content)
        # lock.acquire()
        f = open(basic_data_save_path+inputfile,'a',encoding='utf-8')
        f.write('date,open,close,high,low,volume,amount,amplitude,pct_chg,change,turnover_rate\n')
        f.write('\n'.join(content['data']['klines']))
        f.close()
        df=pd.read_csv(basic_data_save_path+inputfile,encoding='utf-8')
        # df['symbol']=inputfile.split('.')[0]
        # # lock.release()
        df.to_parquet(basic_data_save_path+inputfile)

        # else:
        #     if inputfile in os.listdir(basic_data_save_path):
        #         os.remove(basic_data_save_path+inputfile)
    # except:
    #     if inputfile in os.listdir(basic_data_save_path):
    #         os.remove(basic_data_save_path+inputfile)
    #     print(inputfile)
    #     print('error')
def callback(result):
    # 在回调函数中关闭和加入进程池
    global lock
    with lock:
        print("Process finished with result:", result)
def download_basic_data_all(data_path,debug=False):
    '''
    下载所有原始数据
    :data_path: 存放地址
    :debug: 是否开启调试模式（调试模式下单进程运行）
    '''
    df = ak.stock_info_a_code_name()['code']
    codelist = list(df)
    codelist = [code + ',' + data_path for code in codelist]

    if debug:
        for code in tqdm(codelist, desc="Downloading (Debug Mode)"):
            download_basic_data(code)
    else:
        lock = Lock()
        with Pool(16, initializer=init, initargs=(lock,)) as pool:
            with tqdm(total=len(codelist), desc="Downloading") as pbar:
                for _ in pool.imap_unordered(download_basic_data, codelist):
                    pbar.update(1)

    print("All processes are finished.")
    deleteNullFile(data_path)
    print('DownloadComplete')
if __name__ == '__main__':

    if 'basic_data' not in os.listdir('./'):
        os.mkdir('./'+'basic_data')
    basic_data_save_path='./'+'basic_data/'
    delet(basic_data_save_path)
    download_basic_data_all(basic_data_save_path,debug=False)
    print(len(os.listdir(basic_data_save_path)))

#python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume
#python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir ~/.qlib/csv_data/my_data --method parse_instruments
#python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/my_data --method parse_instruments
