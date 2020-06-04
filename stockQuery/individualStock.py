import baostock as bs
import pandas as pd


def query_day_k_data(code):

    rs = bs.query_history_k_data_plus(code,
                                      "date,code,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                                      start_date='2010-01-01',
                                      frequency="d",
                                      adjustflag="3")
    if rs.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
        return

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result.to_csv(code+"_day_k_data.csv", index=False)
    print(result)

    return result
