import baostock as bs
from utils.csv_utils import write_individual, get_csv_latest_date, get_next_day_str, get_today_str, individual_name, \
    index_name, write_index, save_stock_code_to_csv


def query_individual_day_k_data(code, start="2000-01-01", append=True):
    if append:
        current_end = get_csv_latest_date(individual_name(code))

        if current_end == 0:
            append = False
        else:
            if current_end == get_today_str():
                print("individual stock [", code, "] is already latest")
                return
            else:
                start = get_next_day_str(current_end)

    rs = bs.query_history_k_data_plus(code,
                                      "date,"
                                      "code,"
                                      "open,"
                                      "high,"
                                      "low,"
                                      "close,"
                                      "preclose,"
                                      "volume,"
                                      "amount,"
                                      "turn,"
                                      "tradestatus,"
                                      "pctChg,"
                                      "peTTM,"
                                      "pbMRQ,"
                                      "psTTM,"
                                      "pcfNcfTTM",
                                      start_date=start,
                                      frequency="d",
                                      adjustflag="3"
                                      )
    if rs.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond error_msg:' + rs.error_msg)
        return

    result = write_individual(code, rs, append=append)
    if append:
        print("individual stock [", code, "] is updated")
    else:
        print("individual stock [", code, "] is fully updated")

    return result


def query_index_day_k_data(code, start="2000-01-01", append=True):
    if append:
        current_end = get_csv_latest_date(index_name(code))

        if current_end == 0:
            append = False
        else:
            if current_end == get_today_str():
                print("index [", code, "] is already latest")
                return
            else:
                start = get_next_day_str(current_end)

    rs = bs.query_history_k_data_plus(code,
                                      "date,"
                                      "code,"
                                      "open,"
                                      "high,"
                                      "low,"
                                      "close,"
                                      "preclose,"
                                      "volume,"
                                      "amount,"
                                      "pctChg",
                                      start_date=start,
                                      frequency="d",
                                      adjustflag="3"
                                      )
    if rs.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond error_msg:' + rs.error_msg)
        return

    result = write_index(code, rs, append=append)
    if append:
        print("index [", code, "] is updated")
    else:
        print("index [", code, "] is fully updated")

    return result


def query_stock_code():
    rs = bs.query_zz500_stocks()
    save_stock_code_to_csv(rs, "zz500")

    rs = bs.query_sz50_stocks()
    save_stock_code_to_csv(rs, "sz50")

    rs = bs.query_hs300_stocks()
    save_stock_code_to_csv(rs, "hs300")

    rs = bs.query_stock_industry()
    save_stock_code_to_csv(rs, "all")
