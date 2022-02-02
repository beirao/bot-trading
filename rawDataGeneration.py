# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:48:50 2021

@author: thoma
"""

from binance.client import Client
import numpy as np

config = eval(open("api.config").read())
api_key = config['api_key']
secret_key = config['secret_key']

##initialisation Binance

client = Client(api_key, secret_key)
prices = client.get_all_tickers()

timeFrame = '1h'
fin = "30 Sep 2021"


klinesETH = client.get_historical_klines("ETHUSDT", timeFrame, "14 Sep 2017", fin)
klinesBTC = client.get_historical_klines("BTCUSDT", timeFrame, "14 Sep 2017", fin)
klinesLTC = client.get_historical_klines("LTCUSDT", timeFrame, "14 Sep 2017", fin)

k0 = klinesETH + klinesBTC + klinesLTC

klines1INCH = client.get_historical_klines("1INCHUSDT", timeFrame, "28 Dec 2020", fin)
klinesAAVE = client.get_historical_klines("AAVEUSDT", timeFrame, "19 Oct 2020", fin)
klinesADA = client.get_historical_klines("ADAUSDT", timeFrame, "07 May 2018", fin)
klinesALGO = client.get_historical_klines("ALGOUSDT", timeFrame, "14 Sep 2019", fin)
klinesALPHA = client.get_historical_klines("ALPHAUSDT", timeFrame, "14 Oct 2020", fin)
klinesANKR = client.get_historical_klines("ANKRUSDT", timeFrame, "14 Sep 2019", fin)
klinesARPA = client.get_historical_klines("ARPAUSDT", timeFrame, "14 Dec 2019", fin)
klinesATOM = client.get_historical_klines("ATOMUSDT", timeFrame, "14 Jun 2019", fin)
klinesAVAX = client.get_historical_klines("AVAXUSDT", timeFrame, "14 Oct 2020", fin)
klinesBAND = client.get_historical_klines("BANDUSDT", timeFrame, "14 Nov 2019", fin)

k1 = klines1INCH + klinesAAVE + klinesADA + klinesALGO + klinesALPHA + klinesALPHA + klinesATOM + klinesARPA + klinesANKR + klinesAVAX + klinesBAND
#
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#
#k2 =
#
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#
#k3 =
#
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#klines = client.get_historical_klines("USDT", timeFrame, "14 Sep 2019", fin)
#
#k4 =


kTrain = k0 + k1

klinesBAT = client.get_historical_klines("BATUSDT", timeFrame, "14 Apr 2019", fin)
klinesDIA = client.get_historical_klines("DIAUSDT", timeFrame, "14 Sep 2020", fin)
klinesFET = client.get_historical_klines("FETUSDT", timeFrame, "14 Mar 2019", fin)
klinesLUNA = client.get_historical_klines("LUNAUSDT", timeFrame, "14 Sep 2020", fin)
klinesXMR = client.get_historical_klines("XMRUSDT", timeFrame, "14 Apr 2019", fin)
klinesNEAR = client.get_historical_klines("NEARUSDT", timeFrame, "14 Nov 2020", fin)
klinesSC = client.get_historical_klines("SCUSDT", timeFrame, "14 Aug 2020", fin)
klinesSUSHI = client.get_historical_klines("SUSHIUSDT", timeFrame, "14 Sep 2020", fin)
klinesSXP = client.get_historical_klines("SXPUSDT", timeFrame, "14 Sep 2020", fin)
klinesWAN = client.get_historical_klines("WANUSDT", timeFrame, "14 Sep 2019", fin)

kTest = klinesNEAR + klinesDIA + klinesFET + klinesLUNA + klinesXMR + klinesBAT + klinesSC + klinesSUSHI + klinesWAN + klinesSXP
kall = kTrain + kTest

kall = np.array(kall)
kTrain = np.array(kTrain)
kTest = np.array(kTest)
klinesBAT = np.array(klinesBAT)
klinesBTC = np.array(klinesBTC)

np.savetxt('rawData/rawAllData.csv', kall, fmt='%s')
np.savetxt('rawData/rawTestBat.csv', klinesBAT, fmt='%s')
np.savetxt('rawData/rawTestBtc.csv', klinesBTC, fmt='%s')




































