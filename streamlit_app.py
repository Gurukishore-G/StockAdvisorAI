import streamlit as st
from portfoliomanager import InvestmentAdvisor, format_currency, format_percentage

st.set_page_config(page_title="StockAdvisor AI", layout="wide")

st.title("📈 GenAI-Powered Financial Advisor")
st.write("Analyze stocks, build portfolios, and get AI-powered recommendations.")

advisor = InvestmentAdvisor(initial_capital=50000)

# Sidebar
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose an action", ["Stock Analysis", "Portfolio Summary", "Market Outlook"])

if mode == "Stock Analysis":
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL or RELIANCE)", "AAPL")
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            analysis, msg = advisor.analyze_stock(ticker.upper())
        if analysis:
            st.subheader(f"📊 Analysis for {ticker.upper()}")
            st.metric("Current Price", format_currency(analysis["current_price"]))
            st.metric("Recommendation", analysis["recommendation"]["action"])
            st.write("**Rationale:**", analysis["recommendation"]["rationale"])
            
            with st.expander("📌 Technical Signals"):
                for k, v in analysis["technical"]["signals"].items():
                    st.write(f"- {k}: {v}")
                    
            with st.expander("📉 Support & Resistance"):
                st.write("Support:", [format_currency(x) for x in analysis["technical"]["support_levels"]])
                st.write("Resistance:", [format_currency(x) for x in analysis["technical"]["resistance_levels"]])
                
            with st.expander("🗞 News & Sentiment"):
                st.write(f"Average Sentiment: {analysis['news']['avg_sentiment']:.2f} ({analysis['news']['sentiment_label']})")
                for article in analysis['news']['articles'][:3]:
                    st.write(f"- [{article['title']}]({article['link']}) — *{article['publisher']}*")

            if analysis['prediction']:
                st.subheader("🔮 AI Price Prediction")
                st.write(f"Predicted Direction: **{analysis['prediction']['predicted_direction']}**")
                st.write(f"Predicted Change: {format_percentage(analysis['prediction']['predicted_change'] * 100)}")
                st.write(f"Confidence: {format_percentage(analysis['prediction']['confidence'] * 100)}")

elif mode == "Portfolio Summary":
    st.subheader("📦 Portfolio Overview")
    port = advisor.build_portfolio(risk_level='moderate', num_stocks=5)[0]
    st.metric("Portfolio Value", format_currency(port['portfolio_value']))
    st.metric("Cash Remaining", format_currency(port['cash_remaining']))
    
    with st.expander("🔐 Executed Trades"):
        for trade in port['trades']:
            st.write(f"{trade['ticker']}: {trade['message']}")
    
    with st.expander("📊 Current Positions"):
        for pos in port['positions']:
            st.write(f"{pos['ticker']} | Shares: {pos['shares']} | P/L: {format_currency(pos['unrealized_pl'])} ({format_percentage(pos['unrealized_pl_pct'])})")

elif mode == "Market Outlook":
    st.subheader("🌎 AI Market Outlook")
    outlook = advisor.generate_market_outlook()
    st.metric("Direction", outlook["market_direction"])
    st.write(outlook["outlook_summary"])
    st.write("Top Performing Sectors:")
    for sector, ret in outlook["top_sectors"]:
        st.write(f"- {sector}: {format_percentage(ret)}")
