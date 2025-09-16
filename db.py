import sqlite3
import pandas as pd

# Connect to the database file
conn = sqlite3.connect(r"C:\Users\yniki\Downloads\Alzehimers\patients.db")

# Query all records
df = pd.read_sql_query("SELECT * FROM patient_records", conn)

# Close connection
conn.close()

# Display
print(df)
