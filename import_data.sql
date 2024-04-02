CREATE TABLE stock_data (
    Date DATE,
    Open NUMERIC,
    High NUMERIC,
    Low NUMERIC,
    Close NUMERIC,
    Adj_Close NUMERIC,
    Volume BIGINT,
    PositiveScore NUMERIC,
    NegativeScore NUMERIC,
    NeutralScore NUMERIC,
    Ticker VARCHAR(10)
);
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/AAPL_combined.csv' WITH CSV HEADER; -- Ticker: AAPL
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/AMD_combined.csv' WITH CSV HEADER; -- Ticker: AMD
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/AMZN_combined.csv' WITH CSV HEADER; -- Ticker: AMZN
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/COST_combined.csv' WITH CSV HEADER; -- Ticker: COST
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/GOOGL_combined.csv' WITH CSV HEADER; -- Ticker: GOOGL
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/GOOG_combined.csv' WITH CSV HEADER; -- Ticker: GOOG
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/HD_combined.csv' WITH CSV HEADER; -- Ticker: HD
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/HSBC_combined.csv' WITH CSV HEADER; -- Ticker: HSBC
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/INTC_combined.csv' WITH CSV HEADER; -- Ticker: INTC
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/JNJ_combined.csv' WITH CSV HEADER; -- Ticker: JNJ
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/JPM_combined.csv' WITH CSV HEADER; -- Ticker: JPM
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/MA_combined.csv' WITH CSV HEADER; -- Ticker: MA
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/META_combined.csv' WITH CSV HEADER; -- Ticker: META
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/MSFT_combined.csv' WITH CSV HEADER; -- Ticker: MSFT
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/NVDA_combined.csv' WITH CSV HEADER; -- Ticker: NVDA
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/PEP_combined.csv' WITH CSV HEADER; -- Ticker: PEP
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/PG_combined.csv' WITH CSV HEADER; -- Ticker: PG
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/TSLA_combined.csv' WITH CSV HEADER; -- Ticker: TSLA
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/V_combined.csv' WITH CSV HEADER; -- Ticker: V
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/WMT_combined.csv' WITH CSV HEADER; -- Ticker: WMT
\COPY stock_data FROM 'D:/Project-FullStack/TaskAutomation/Data/XOM_combined.csv' WITH CSV HEADER; -- Ticker: XOM
