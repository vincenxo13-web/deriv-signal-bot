#!/bin/bash
python main.py &
streamlit run dashboard.py --server.address=0.0.0.0 --server.port=$PORT
