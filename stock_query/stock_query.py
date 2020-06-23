import baostock as bs

from csv_utils import write_individual, get_csv_latest_date, get_next_day_str, getTodayStr


def query_day_k_data(code, start="2000-01-01", append=True):
    if append:
        current_end = get_csv_latest_date(code)

        if current_end == 0:
            append = False
        else:
            if current_end == getTodayStr():
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

    return result
