# MT5 Relay Server

Bridges the gap between this Linux-based bot and your Windows MetaTrader5 terminal.

## Setup (on your Windows PC)

### 1. Install dependencies
```bash
pip install flask MetaTrader5
```

### 2. Open MetaTrader5
Log in to your TradersWay demo account (1120409).

### 3. Edit the token
Open `relay_server.py` and change `RELAY_TOKEN` to match the `MT5_RELAY_TOKEN` in your `.env` file.

### 4. Run the relay
```bash
python relay_server.py
```
You should see:
```
✅ Connected to MT5 | Account: 1120409 | Balance: 1000.0
🌐 Relay server starting on http://0.0.0.0:5555
```

### 5. Expose with ngrok
In a separate terminal:
```bash
ngrok http 5555
```
Copy the `https://xxxx.ngrok-free.app` URL.

### 6. Update `.env` in this Codespace
```
MT5_RELAY_URL=https://xxxx.ngrok-free.app
MT5_RELAY_TOKEN=change-me-to-a-secret
```

### 7. Restart the bot
The bot will now connect through the relay and place real orders on your MT5 demo account.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ping` | Health check |
| GET | `/account` | Account info |
| GET | `/balance` | Current balance |
| GET | `/candles?pair=EURUSD&timeframe=5&count=100` | OHLCV data |
| GET | `/price?pair=EURUSD` | Latest bid/ask |
| GET | `/positions` | Open positions |
| POST | `/order` | Place order |
| POST | `/close` | Close position |
