from src.broker.rithmic_connector import RithmicConnector as R
f = R._position_qty_from_broker_data

def test_flat_with_working_buy_orders_is_zero():
    # Flat after the 2-lot short closed; leftover BUY TP 2 still working.
    d = {"buy_qty": 2, "sell_qty": 0, "order_buy_qty": 2, "fill_buy_qty": 2,
         "fill_sell_qty": 2, "net_quantity": 0}
    assert f(d) == 0

def test_fills_only_no_net_field():
    assert f({"buy_qty": 2, "sell_qty": 0, "fill_buy_qty": 0, "fill_sell_qty": 0}) == 0
    assert f({"buy_qty": 4, "sell_qty": 2, "fill_buy_qty": 0, "fill_sell_qty": 2}) == -2

def test_real_short_and_legacy():
    assert f({"net_quantity": -2, "buy_qty": 2, "sell_qty": 2}) == -2
    assert f({"buy_qty": 3, "sell_qty": 1}) == 2
