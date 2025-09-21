# 📊 Portfolio Optimizer Backend

This is the backend service for the **Portfolio Optimizer MVP**.  
It provides an API for optimizing stock portfolios and generating plots (allocation pie chart and efficient frontier).

---

## 🚀 Features
- REST API built with **Flask**  
- Portfolio optimization logic in **Python**  
- Generates plots with **Matplotlib**  
- Returns results + plots as **JSON** (plots serialized as base64 strings)  
- Ready to connect to a React frontend  

---

## 📂 Project Structure
```bash
backend/
│
├── app.py                  # Entry point (runs Flask server)
├── requirements.txt        # Python dependencies
│
├── api/
│   └── routes.py           # API endpoints (/api/optimize)
│
├── portfolio/
│   └── optimizer.py        # Core portfolio logic + plots
│
├── utils/
│   └── serializer.py       # Convert matplotlib figures → base64
│
└── tests/                  # (Optional) Unit tests
```

Installation
```bash
git clone https://github.com/your-username/portfolio-optimizer.git
cd portfolio-optimizer/backend
```

2. Create virtual enviroment 
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
Running the server
```bash
python app.py
```
The server will start at 
```
http://127.0.0.1:5000
```

📡 API Usage
Endpoint
```bash
POST /api/optimize
```

Request JSON
```{
  "stocks": ["AAPL", "TSLA", "MSFT"]
}
```

Response JSON
```
{
  "results": {
    "expected_return": 0.14,
    "volatility": 0.15,
    "weights": {
      "AAPL": 0.33,
      "TSLA": 0.33,
      "MSFT": 0.33
    }
  },
  "plots": {
    "pie_chart": "<base64 string>",
    "efficient_frontier": "<base64 string>"
  }
}
```

Frontend Integration
```
<img src={`data:image/png;base64,${data.plots.pie_chart}`} alt="Pie Chart" />
<img src={`data:image/png;base64,${data.plots.efficient_frontier}`} alt="Efficient Frontier" />
```

📌 Notes

This is a development server (debug mode).

For production, use a WSGI server like Gunicorn or uWSGI.

The optimization logic currently uses placeholder values; you can replace it with real finance calculations (expected returns, covariance, Sharpe ratio, etc.).
