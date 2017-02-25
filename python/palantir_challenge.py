def findFraudolentTraders(datafeed):
    flagged_trades = set()
    trades = dict()
    current_price = None
    for feed in datafeed:
        vals = feed.split("|")
        day = int(vals[0])
        if len(vals) == 2:
            current_price = int(vals[1])
            for x in range(day - 3, day):
                if x in trades:
                    for (trader_name, isBuy, price, amount) in trades[x]:
                        if (x, trader_name) in flagged_trades:
                            continue
                        if isBuy:
                            fraudolent = (current_price - price) * \
                                amount >= 500000
                        else:
                            fraudolent = (price - current_price) * \
                                amount >= 500000
                        if fraudolent:
                            flagged_trades.add((x, trader_name))
        else:
            trader_name = vals[1]
            isBuy = len(vals[2]) == 3
            amount = int(vals[3])
            if day not in trades:
                trades[day] = []
            trades[day].append((trader_name, isBuy, current_price, amount))
    flagged_trades = sorted(list(flagged_trades))
    return list(map(lambda x: str(x[0]) + "|" + str(x[1]), flagged_trades))


feed2 = """0|20
0|Kristi|SELL|3000
0|Will|BUY|5000
0|Tom|BUY|50000
0|Shilpa|BUY|1500
1|Tom|BUY|1500000
3|25
5|Shilpa|SELL|1500
8|Kristi|SELL|600000
9|Shilpa|BUY|500
10|15
11|5
14|Will|BUY|100000
15|Will|BUY|100000
16|Will|BUY|100000
17|25"""

datafeed2 = feed2.split("\n")

print findFraudolentTraders(datafeed2)
