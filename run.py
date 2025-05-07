"""Run script for the Streamlit application."""
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "src/app/main.py", "--server.address", "0.0.0.0"]
    sys.exit(stcli.main()) 
