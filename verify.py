import pandas as pd
import pyarrow.parquet as pq
import os

def calculate_growth_rate(df, parquet_folder, output_file):
    """
    遍历 DataFrame 的每一行，读取对应的 Parquet 文件，计算增长率，并更新 DataFrame。

    Args:
        df: 包含 stock_id 和 date 列的 Pandas DataFrame。
        parquet_folder: 存储 Parquet 文件的文件夹路径。
        output_file: 保存更新后 DataFrame 的 CSV 文件路径。
    """

    results = []  # 用于存储每一行的结果
    if 'true'  in df.columns and df['true'].isnull().sum()==0:
        return

    for index, row in df.iterrows():
        stock_id = row['stock_id']
        date = row['date']
        predict = row['predict']
        name = row['name']

        # 构建 Parquet 文件路径
        parquet_file = os.path.join(parquet_folder, f"{stock_id}.parquet")

        try:
            # 读取 Parquet 文件
            table = pq.read_table(parquet_file, columns=['date', 'open']) # 假设你的价格数据列名为 'open'
            stock_df = table.to_pandas()

            # 将日期列转换为 datetime 类型
            stock_df['date'] = pd.to_datetime(stock_df['date'])

            # 找到对应日期的索引
            target_date_index = stock_df[stock_df['date'] == pd.to_datetime(date)].index[0]

            # 计算增长率
            if target_date_index + 2 < len(stock_df):
                price_plus_1 = stock_df.iloc[target_date_index + 1]['open']
                price_plus_2 = stock_df.iloc[target_date_index + 2]['open']
                growth_rate = (price_plus_2 - price_plus_1) / price_plus_1
            else:
                growth_rate = None  # 如果没有足够的数据，则将增长率设为 None

        except (FileNotFoundError, IndexError, KeyError):
            # 处理文件不存在、索引错误或键错误的情况
            growth_rate = None

        # 将结果添加到列表中
        results.append({'date': date,'stock_id': stock_id,"name":name, 'predict':predict, 'true': growth_rate})

    # 将结果列表转换为 DataFrame
    result_df = pd.DataFrame(results)

    # 保存 DataFrame 到 CSV 文件
    result_df.to_csv(output_file, index=False)
    print(f"DataFrame 已保存到 {output_file}")

# 示例用法
# 假设你的 DataFrame 名为 'data'，Parquet 文件在 'parquet_data' 文件夹中，
# 并且你想将结果保存到 'output.csv' 文件中
for veridy_name in os.listdir('./predict_result/'):
    parquet_folder = './basic_data'  # 替换为你的 Parquet 文件所在的文件夹
    # output_file = 'output.csv'  # 替换为你想要保存结果的文件名
    data=pd.read_csv('./predict_result/'+veridy_name,encoding='utf-8')
    calculate_growth_rate(data, parquet_folder, './predict_result/'+veridy_name)
