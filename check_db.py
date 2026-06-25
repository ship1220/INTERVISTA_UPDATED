import sqlite3

conn = sqlite3.connect("jobprepmate.db")
cursor = conn.cursor()

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Show users
cursor.execute("SELECT * FROM users")
print("Users:", cursor.fetchall())

conn.close()