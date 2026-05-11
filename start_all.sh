#!/bin/bash
set -e
python main.py &
streamlit run dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-8080}
