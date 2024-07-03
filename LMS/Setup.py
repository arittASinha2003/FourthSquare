import mysql.connector as mq
import os
from typing import Optional
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Function to establish database connection with error handling
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
def initialize_database() -> None:
    """
    Initializes the database by creating necessary tables if they don't exist.
    """
    con: Optional[mq.MySQLConnection] = get_database_connection()
    if con:
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
        con.close()

# Main function
def main() -> None:
    """
    Main function to initialize the database.
    """
    initialize_database()
    print("\nSetup Successful :)\n")

if __name__ == "__main__":
    main()
