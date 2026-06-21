import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from database import DATABASE_URL, MarketData, TradeStore, PortfolioState
from config import DASH_HOST, DASH_PORT, logger

# Database connection helper
engine = create_engine(DATABASE_URL)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Quantitative Decision Engine"
)

# Premium Custom CSS Styles
styles = {
    "body": {
        "backgroundColor": "#0b0f19",
        "color": "#f1f5f9",
        "fontFamily": "'Outfit', 'Inter', -apple-system, sans-serif",
        "margin": "0",
        "padding": "0"
    },
    "header": {
        "backgroundColor": "#111827",
        "padding": "20px 30px",
        "borderBottom": "1px solid #1f2937",
        "display": "flex",
        "justifyContent": "between",
        "alignItems": "center"
    },
    "card": {
        "backgroundColor": "#1e293b",
        "borderRadius": "10px",
        "border": "1px solid #334155",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)"
    },
    "kpi_value": {
        "fontSize": "28px",
        "fontWeight": "bold",
        "color": "#60a5fa",
        "marginTop": "5px"
    },
    "kpi_label": {
        "fontSize": "12px",
        "color": "#94a3b8",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em"
    }
}

app.layout = html.Div(
    style=styles["body"],
    children=[
        # Interval for real-time updates
        dcc.Interval(id="update-interval", interval=2000, n_intervals=0),
        
        # Header Area
        html.Div(
            style=styles["header"],
            children=[
                html.Div([
                    html.H1(
                        "QUANTITATIVE DECISION ENGINE",
                        style={"margin": "0", "fontSize": "24px", "fontWeight": "800", "letterSpacing": "0.05em", "color": "#f8fafc"}
                    ),
                    html.Span(
                        "Live Time-Series Intelligence & Risk Controller",
                        style={"color": "#94a3b8", "fontSize": "12px", "fontWeight": "500"}
                    )
                ]),
                # Status badges
                html.Div(
                    style={"display": "flex", "gap": "15px", "alignItems": "center"},
                    children=[
                        html.Div(
                            id="simulator-status",
                            children="SIMULATOR ACTIVE",
                            style={
                                "padding": "5px 12px", "borderRadius": "20px", "fontSize": "11px",
                                "fontWeight": "600", "backgroundColor": "#064e3b", "color": "#34d399",
                                "border": "1px solid #059669"
                            }
                        ),
                        html.Div(
                            id="model-status",
                            children="XGBOOST LOADED",
                            style={
                                "padding": "5px 12px", "borderRadius": "20px", "fontSize": "11px",
                                "fontWeight": "600", "backgroundColor": "#1e3a8a", "color": "#93c5fd",
                                "border": "1px solid #2563eb"
                            }
                        )
                    ]
                )
            ]
        ),
        
        # Grid Dashboard Layout
        html.Div(
            style={"padding": "30px", "maxWidth": "1600px", "margin": "0 auto"},
            children=[
                # KPI Row
                html.Div(
                    className="row",
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        html.Div(style=styles["card"], children=[
                            html.Div("Total Return", style=styles["kpi_label"]),
                            html.Div(id="kpi-return", style=styles["kpi_value"])
                        ]),
                        html.Div(style=styles["card"], children=[
                            html.Div("Sharpe Ratio", style=styles["kpi_label"]),
                            html.Div(id="kpi-sharpe", style=styles["kpi_value"])
                        ]),
                        html.Div(style=styles["card"], children=[
                            html.Div("Max Drawdown", style=styles["kpi_label"]),
                            html.Div(id="kpi-drawdown", style=styles["kpi_value"])
                        ]),
                        html.Div(style=styles["card"], children=[
                            html.Div("Win Rate", style=styles["kpi_label"]),
                            html.Div(id="kpi-winrate", style=styles["kpi_value"])
                        ]),
                        html.Div(style=styles["card"], children=[
                            html.Div("Profit Factor", style=styles["kpi_label"]),
                            html.Div(id="kpi-profitfactor", style=styles["kpi_value"])
                        ])
                    ]
                ),
                
                # Middle Row (Charts)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        # Price Chart Card
                        html.Div(
                            style=styles["card"],
                            children=[
                                html.H3("Market Price & Trade Executions", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#f1f5f9"}),
                                dcc.Graph(id="price-chart", style={"height": "400px"})
                            ]
                        ),
                        # Model Probabilities Gauge
                        html.Div(
                            style=styles["card"],
                            children=[
                                html.H3("Model Confidence Forecast", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#f1f5f9"}),
                                dcc.Graph(id="confidence-chart", style={"height": "400px"})
                            ]
                        )
                    ]
                ),
                
                # Bottom Row (Equity & Ledger)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
                    children=[
                        # Equity Curve Card
                        html.Div(
                            style=styles["card"],
                            children=[
                                html.H3("Strategy Equity Performance", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#f1f5f9"}),
                                dcc.Graph(id="equity-chart", style={"height": "350px"})
                            ]
                        ),
                        # Trade Ledger Card
                        html.Div(
                            style=styles["card"],
                            children=[
                                html.H3("Recent Trade Executions Ledger", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#f1f5f9"}),
                                html.Div(
                                    id="trade-table-container",
                                    style={"height": "350px", "overflowY": "auto"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Callback to refresh all database metrics and charts
@app.callback(
    [
        Output("kpi-return", "children"),
        Output("kpi-sharpe", "children"),
        Output("kpi-drawdown", "children"),
        Output("kpi-winrate", "children"),
        Output("kpi-profitfactor", "children"),
        Output("price-chart", "figure"),
        Output("equity-chart", "figure"),
        Output("confidence-chart", "figure"),
        Output("trade-table-container", "children")
    ],
    [Input("update-interval", "n_intervals")]
)
def update_dashboard(n):
    # 1. Read Tables
    try:
        df_md = pd.read_sql("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 300", con=engine)
        df_trades = pd.read_sql("SELECT * FROM trade_store ORDER BY timestamp DESC", con=engine)
        df_port = pd.read_sql("SELECT * FROM portfolio_state ORDER BY timestamp ASC", con=engine)
    except Exception as e:
        logger.error(f"Error loading dashboard DB data: {e}")
        # Return empty charts
        return "-", "-", "-", "-", "-", go.Figure(), go.Figure(), go.Figure(), html.Div("No active session data.")

    # Handle empty tables
    if df_md.empty:
        return "-", "-", "-", "-", "-", go.Figure(), go.Figure(), go.Figure(), html.Div("Waiting for data ingestion...")

    # Sort market data chronologically for graphing
    df_md = df_md.sort_values("timestamp")

    # 2. Compute KPIs
    total_return_str = "0.0%"
    sharpe_str = "0.00"
    max_dd_str = "0.0%"
    winrate_str = "0.0%"
    profitfactor_str = "0.00"
    
    if not df_port.empty:
        initial_eq = df_port.iloc[0]["equity"]
        final_eq = df_port.iloc[-1]["equity"]
        ret = (final_eq - initial_eq) / initial_eq
        total_return_str = f"{ret * 100:.2f}%"
        
        # Calculate Sharpe Ratio from equity curve
        eq_returns = df_port["equity"].pct_change().dropna()
        if eq_returns.std() > 0:
            sharpe = (eq_returns.mean() / eq_returns.std()) * np.sqrt(252)
            sharpe_str = f"{sharpe:.2f}"
            
        max_dd = df_port["drawdown"].max()
        max_dd_str = f"{max_dd * 100:.2f}%"

    closed_sells = df_trades[df_trades["side"] == "SELL"] if not df_trades.empty else pd.DataFrame()
    if not closed_sells.empty:
        wins = closed_sells[closed_sells["pnl"] > 0]
        winrate = len(wins) / len(closed_sells)
        winrate_str = f"{winrate * 100:.1f}%"
        
        gross_profit = closed_sells[closed_sells["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(closed_sells[closed_sells["pnl"] <= 0]["pnl"].sum())
        
        profitfactor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        profitfactor_str = f"{profitfactor:.2f}"

    # 3. Price Chart with Trades
    symbol = df_md.iloc[-1]["symbol"]
    price_fig = go.Figure()
    
    # Check if we are drawing bars or lines
    if "open" in df_md.columns and df_md["open"].notna().any():
        # Candlestick
        price_fig.add_trace(go.Candlestick(
            x=df_md["timestamp"],
            open=df_md["open"],
            high=df_md["high"],
            low=df_md["low"],
            close=df_md["close"],
            name=symbol,
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        ))
    else:
        # Tick line
        price_fig.add_trace(go.Scatter(
            x=df_md["timestamp"], y=df_md["close"],
            mode="lines", name=symbol, line=dict(color="#60a5fa", width=2)
        ))
        
    # Overlay Buy/Sell trades
    if not df_trades.empty:
        buys = df_trades[df_trades["side"] == "BUY"]
        sells = df_trades[df_trades["side"] == "SELL"]
        
        price_fig.add_trace(go.Scatter(
            x=buys["timestamp"], y=buys["price"],
            mode="markers", name="BUY Order",
            marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(width=1, color="white"))
        ))
        price_fig.add_trace(go.Scatter(
            x=sells["timestamp"], y=sells["price"],
            mode="markers", name="SELL Order",
            marker=dict(symbol="triangle-down", size=14, color="#ef4444", line=dict(width=1, color="white"))
        ))
        
    price_fig.update_layout(
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#334155", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#334155"),
        showlegend=True
    )

    # 4. Equity Chart
    equity_fig = go.Figure()
    if not df_port.empty:
        equity_fig.add_trace(go.Scatter(
            x=df_port["timestamp"], y=df_port["equity"],
            mode="lines", fill="tozeroy", name="Net Equity",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59, 130, 246, 0.1)"
        ))
    equity_fig.update_layout(
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        showlegend=False
    )

    # 5. Model Confidence gauge/donut chart
    # We will simulate/generate a default distribution if no signals table exists,
    # otherwise we display the latest prediction confidence.
    conf_fig = go.Figure()
    
    # Try to load signal probabilities
    try:
        df_sig = pd.read_sql("SELECT * FROM signal_store ORDER BY timestamp DESC LIMIT 1", con=engine)
    except Exception:
        df_sig = pd.DataFrame()
        
    if not df_sig.empty:
        prob_down = df_sig.iloc[0]["prob_down"]
        prob_neut = df_sig.iloc[0]["prob_neutral"]
        prob_up = df_sig.iloc[0]["prob_up"]
        pred = df_sig.iloc[0]["prediction"]
        pred_label = "BUY (UP)" if pred == 1 else "SELL (DOWN)" if pred == -1 else "HOLD (NEUTRAL)"
    else:
        # Defaults
        prob_down, prob_neut, prob_up = 0.2, 0.5, 0.3
        pred_label = "HOLD (NEUTRAL)"

    conf_fig.add_trace(go.Pie(
        labels=["DOWN (Sell)", "NEUTRAL (Hold)", "UP (Buy)"],
        values=[prob_down, prob_neut, prob_up],
        hole=0.6,
        marker=dict(colors=["#ef4444", "#64748b", "#10b981"]),
        hoverinfo="label+percent"
    ))
    
    conf_fig.update_layout(
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font=dict(color="#94a3b8"),
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[dict(text=pred_label, x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color="#f8fafc", weight="bold"))],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    # 6. Trade Ledger Table
    if df_trades.empty:
        table_child = html.Div("No trades recorded yet.", style={"color": "#94a3b8", "padding": "20px", "textAlign": "center"})
    else:
        # Formulate columns
        table_rows = []
        # Header
        table_rows.append(html.Tr([
            html.Th("Timestamp", style={"textAlign": "left", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Symbol", style={"textAlign": "left", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Side", style={"textAlign": "left", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Price", style={"textAlign": "right", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Qty", style={"textAlign": "right", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Net PnL", style={"textAlign": "right", "padding": "10px", "borderBottom": "2px solid #334155"}),
            html.Th("Status", style={"textAlign": "center", "padding": "10px", "borderBottom": "2px solid #334155"})
        ]))
        
        for _, row in df_trades.head(50).iterrows():
            pnl_color = "#10b981" if row["pnl"] > 0 else "#ef4444" if row["pnl"] < 0 else "#94a3b8"
            pnl_val = f"${row['pnl']:,.2f}" if row["side"] == "SELL" else "-"
            side_color = "#10b981" if row["side"] == "BUY" else "#ef4444"
            
            table_rows.append(html.Tr([
                html.Td(row["timestamp"], style={"padding": "8px 10px", "borderBottom": "1px solid #1e293b", "fontSize": "12px"}),
                html.Td(row["symbol"], style={"padding": "8px 10px", "borderBottom": "1px solid #1e293b"}),
                html.Td(row["side"], style={"padding": "8px 10px", "borderBottom": "1px solid #1e293b", "color": side_color, "fontWeight": "bold"}),
                html.Td(f"${row['price']:,.2f}", style={"textAlign": "right", "padding": "8px 10px", "borderBottom": "1px solid #1e293b"}),
                html.Td(f"{row['qty']:.4f}", style={"textAlign": "right", "padding": "8px 10px", "borderBottom": "1px solid #1e293b"}),
                html.Td(pnl_val, style={"textAlign": "right", "padding": "8px 10px", "borderBottom": "1px solid #1e293b", "color": pnl_color, "fontWeight": "bold"}),
                html.Td(row["status"], style={"textAlign": "center", "padding": "8px 10px", "borderBottom": "1px solid #1e293b", "fontSize": "11px"})
            ]))
            
        table_child = html.Table(table_rows, style={"width": "100%", "borderCollapse": "collapse", "color": "#cbd5e1"})

    return total_return_str, sharpe_str, max_dd_str, winrate_str, profitfactor_str, price_fig, equity_fig, conf_fig, table_child


if __name__ == "__main__":
    app.run_server(host=DASH_HOST, port=DASH_PORT, debug=True)
