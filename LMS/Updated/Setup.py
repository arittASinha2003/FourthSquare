import mysql.connector as mq
import os
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    filename = 'LMS.log',
    level = logging.INFO,
    format = '%(asctime)s - %(message)s',
    datefmt = '%d-%m-%Y %H:%M:%S'
)

# Header Text for Log File
header_text = """
--------------------------------------------------------------------------------------
                                        LOG FILE
--------------------------------------------------------------------------------------
                                Library Management System
--------------------------------------------------------------------------------------

    Date    Time                            Action
--------------------------------------------------------------------------------------

"""

# Check if the log file is empty
if os.path.getsize('LMS.log') == 0:
    # Write the header text to the log file
    with open('LMS.log', 'a') as log_file:
        log_file.write(header_text)

# Decorator for logging
def log_action(action):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"Action Started: {action}")
            result = func(*args, **kwargs)
            logging.info(f"Action Completed: {action}")
            return result
        return wrapper
    return decorator

# Context Manager for Database Connection
class DatabaseConnection:
    def __enter__(self):
        self.con = get_database_connection()
        return self.con

    def __exit__(self, exc_type, exc_value, traceback):
        if self.con:
            self.con.close()

# Function to establish database connection with error handling
@log_action("Establishing Database Connection")
def get_database_connection() -> Optional[mq.MySQLConnection]:
    """
    Establishes a connection to the MySQL database.
    """
    try:
        con: mq.MySQLConnection = mq.connect(
            host = os.getenv('host'), 
            user = os.getenv('user'), 
            password = os.getenv('password'), 
            database = os.getenv('database')
        )
        return con
    except mq.Error as e:
        print("\nError connecting to the database:", e)
        print("\n")
        exit()

# Function to initialize database tables
@log_action("Initializing Database Tables")
def initialize_database() -> None:
    """
    Initializes the database by creating necessary tables if they don't exist.
    """
    with DatabaseConnection() as con:
        cur: mq.cursor.MySQLCursor = con.cursor()

        # USERS TABLE
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                email VARCHAR(50) NOT NULL UNIQUE,
                issued_book_count INT DEFAULT 0
            )
        ''')

        # BOOKS TABLE
        cur.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id INT PRIMARY KEY,
                title VARCHAR(50) NOT NULL,
                author VARCHAR(50) NOT NULL,
                genre VARCHAR(50) NOT NULL,
                available BOOLEAN DEFAULT TRUE,
                user_id INT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # ISSUEDBOOKS TABLE
        cur.execute('''
            CREATE TABLE IF NOT EXISTS issuedbooks (
                sno INT AUTO_INCREMENT PRIMARY KEY,
                book_id INT,
                issued_book_title VARCHAR(50) NOT NULL,
                user_id INT,
                issue_date DATE NOT NULL,
                due_date DATE NOT NULL,
                FOREIGN KEY (book_id) REFERENCES books(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        con.commit()

# Main function
def main() -> None:
    """
    Main function to initialize the database.
    """
    initialize_database()
    print("\nSetup Successful :)\n")

if __name__ == "__main__":
    main()
