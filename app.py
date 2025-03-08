import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the dashboard app
from dashboard.app import main

# Initialize database if it doesn't exist
from cost_tracking.database import init_db

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'costs.db')
if not os.path.exists(db_path):
    print("Initializing database...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    init_db()

# Run the dashboard
if __name__ == "__main__":
    main() 