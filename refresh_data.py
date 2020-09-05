# coding=gbk
from stock_query.stock import prepare_data
from utils.csv_utils import get_all_stocks_code_list, get_stock_code_list_by_industry

append_mode = False

index_list_query = ["sh.000001",
                    "sz.399106",
                    "sh.000016",
                    "sh.000300",
                    "sh.000905",
                    "sz.399001",
                    "sh.000037",
                    "sz.399433",
                    "sh.000952",
                    "sh.000050",
                    "sh.000982",
                    "sh.000029",
                    "sh.000015",
                    "sh.000063",
                    "sh.000011",
                    "sh.000012",
                    "sh.000159",  # 沪股通
                    "sz.399006"   # 创业板指
                    ]

all_stock_list = get_all_stocks_code_list()
prepare_data(all_stock_list, index_list_query, append=append_mode)
