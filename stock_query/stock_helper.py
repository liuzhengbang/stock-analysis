import baostock as bs


def login():
    lg = bs.login()
    if lg.error_code != '0':
        print('login respond error_code:' + lg.error_code)
        print('login respond error_msg:' + lg.error_msg)


def logout():
    bs.logout()
