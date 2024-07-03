import mysql.connector as mq
from tabulate import tabulate
from datetime import datetime, timedelta
from typing import Optional
import re
import os
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

# Function for executing SQL queries with error handling
def execute_query(connection: mq.MySQLConnection, query: str, data=None):
    """
    Executes an SQL query on the database.
    """
    try:
        cursor: mq.cursor.MySQLCursor = connection.cursor()
        if data:
            cursor.execute(query, data)
        else:
            cursor.execute(query)
        connection.commit()
        return cursor
    except mq.Error as e:
        print("Error executing query:", e)
        connection.rollback()
        return None

# Function to validate numeric input with error handling
def validate_numeric_input(prompt: str) -> int:
    """
    Validates numeric input from the user.
    """
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("\nPlease enter a valid number!\n")

# Function to validate email input with error handling
def validate_email() -> str:
    """
    Validates email input from the user.
    """
    while True:
        email = input("Enter Your Email: ")
        pattern = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        if re.match(pattern, email):
            return email
        else:
            print("\nPlease enter a valid email address!\n")

# Function to add a book to the database
def add_book() -> None:
    """
    Adds a book to the database.
    """
    print("\n\t\t\t\tADD BOOK\n")
    bid: int = validate_numeric_input("Enter Book ID: ")
    title: str = input("Enter Title: ")
    author: str = input("Enter Author: ")
    genre: str = input("Enter Genre: ")

    con = get_database_connection()
    if con:
        cur = con.cursor()

        # Check if the book already exists
        cur.execute("SELECT * FROM books WHERE id = %s", (bid,))
        existing_book = cur.fetchone()
        if existing_book:
            print("Book with ID '{}' already exists in the database.".format(bid))
            con.close()
            return

        # Insert the new book into the books table
        query: str = "INSERT INTO books (id, title, author, genre) VALUES (%s, %s, %s, %s)"
        data: tuple = (bid, title, author, genre)
        if execute_query(con, query, data):
            print("\nSuccessfully Added the Book!")
        con.close()

# Function to remove a book from the database
def remove_book() -> None:
    """
    Removes a book from the database.
    """
    print("\n\t\t\t\tREMOVE BOOK\n")
    bid = validate_numeric_input("Enter ID of the Book to Remove: ")

    con = get_database_connection()
    if con:
        cur = con.cursor()

        # Check if the book is issued
        cur.execute("SELECT user_id FROM books WHERE id = %s", (bid,))
        issued_result = cur.fetchone()
        if issued_result:
            if issued_result[0]:
                print("Book with ID '{}' is already issued to user ID: {}".format(bid, issued_result[0]))
                print("Please return the book before removing it.")
                con.close()
                return

        # Check if the book title is present in the books table
        cur.execute("SELECT * FROM books WHERE id = %s", (bid,))
        book_result = cur.fetchone()
        if not book_result:
            print("Book with ID '{}' is not present in the database.".format(bid))
            con.close()
            return

        # Remove the book from the books table
        query: str = "DELETE FROM books WHERE id = %s"
        data: tuple = (bid,)
        if execute_query(con, query, data):
            print("\nSuccessfully Removed the Book!")
        con.close()

# Function to lend a book
def lend_book() -> None:
    """
    Lends a book to a user.
    """
    print("\n\t\t\t\tLEND BOOK\n")
    bid: int = validate_numeric_input("Enter ID of the Book to Lend: ")
    user_id: int = validate_numeric_input("Enter Your ID: ")

    con = get_database_connection()
    if con:
        cur = con.cursor()

        # Check if user exists, if not, register them
        cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user_result = cur.fetchone()
        if not user_result:
            user_name: str = input("Enter Your Name: ")
            user_email: str = validate_email()

            # Insert user into users table
            query: str = "INSERT INTO users (id, name, email, issued_book_count) VALUES (%s, %s, %s, %s)"
            data: tuple = (user_id, user_name, user_email, 0)
            execute_query(con, query, data)
            print("\nUser registered successfully!")

        # Check the number of books issued to the user
        cur.execute("SELECT issued_book_count FROM users WHERE id = %s", (user_id,))
        issued_book_count = cur.fetchone()[0]
        if issued_book_count >= 3:
            print("Maximum 3 books can be lent at a time.")
            con.close()
            return

        # Check if the book title is present in the books table and available
        cur.execute("SELECT * FROM books WHERE id = %s AND available = 1", (bid,))
        book_result = cur.fetchone()
        if not book_result:
            print("Book with ID '{}' is currently not available.".format(bid))
            con.close()
            return

        # Extract the title from the book_result
        title = book_result[1]

        # Update books table
        query: str = "UPDATE books SET available = 0, user_id = %s WHERE id = %s"
        data: tuple = (user_id, bid)
        execute_query(con, query, data)

        # Get issue date and calculate due date
        issue_date = datetime.now()
        due_date = issue_date + timedelta(days=15)

        # Insert into issuedbooks table
        query: str = "INSERT INTO issuedbooks (book_id, issued_book_title, user_id, issue_date, due_date) VALUES (%s, %s, %s, %s, %s)"
        data: tuple = (bid, title, user_id, issue_date, due_date)
        execute_query(con, query, data)

        # Update user's issued book count
        query: str = "UPDATE users SET issued_book_count = issued_book_count + 1 WHERE id = %s"
        data: tuple = (user_id,)
        execute_query(con, query, data)

        print("\nSuccessfully Lent the Book!")
        con.close()

# Function to return a book
def return_book() -> None:
    """
    Returns a book that was previously lent.
    """
    print("\n\t\t\t\tRETURN BOOK\n")
    bid: int = validate_numeric_input("Enter ID of the Book to Return: ")

    con = get_database_connection()
    if con:
        cur = con.cursor()

        # Check if the book title is present in the books table
        cur.execute("SELECT * FROM books WHERE id = %s", (bid,))
        book_result = cur.fetchone()
        if not book_result:
            print("Book with ID '{}' is not registered in the database.".format(bid))
            con.close()
            return

        # Check if the book is issued
        cur.execute("SELECT user_id FROM books WHERE id = %s", (bid,))
        issued_result = cur.fetchone()
        if issued_result:
            if not issued_result[0]:
                print("Book with ID '{}' is not issued to anyone.".format(bid))
                con.close()
                return

        # Update books table
        query: str = "UPDATE books SET available = 1, user_id = NULL WHERE id = %s"
        data: tuple = (bid,)
        execute_query(con, query, data)

        # Update user's issued book count
        query: str = "UPDATE users SET issued_book_count = issued_book_count - 1 WHERE id = (SELECT user_id FROM issuedbooks WHERE book_id = %s)"
        execute_query(con, query, (bid,))

        # Remove entry from issuedbooks table
        query: str = "DELETE FROM issuedbooks WHERE book_id = %s"
        execute_query(con, query, (bid,))

        print("\nSuccessfully Returned the Book!")
        con.close()

# Function to search books by title, author, or genre
def search_books() -> None:
    """
    Searches for books by title, author, or genre.
    """
    while True:
        print("\n\t\t\t\tSEARCH BOOKS\n")
        print("1. Search by Title")
        print("2. Search by Author")
        print("3. Search by Genre")
        print("4. Go Back")

        choice = validate_numeric_input("\nEnter your choice: ")

        if choice == 1:
            title: str = input("\nEnter the title to search: ")
            print("\n")
            search_books_by_title(title)
        elif choice == 2:
            author: str = input("\nEnter the author to search: ")
            print("\n")
            search_books_by_author(author)
        elif choice == 3:
            genre: str = input("\nEnter the genre to search: ")
            print("\n")
            search_books_by_genre(genre)
        elif choice == 4:
            break
        else:
            print("\nInvalid choice! Please select a valid option.")

        choice = validate_numeric_input("\n\nPress 0 To Continue Searching, Any Other Number To Exit...: ")
        if choice != 0:
            break

# Function to search books by title
def search_books_by_title(title: str) -> None:
    """
    Searches for books by title.
    """
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select books by title using a wildcard search
        query = "SELECT ID, Title, Author, Genre FROM books WHERE Title LIKE %s"
        cur.execute(query, ('%' + title + '%',))
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["ID", "Title", "Author", "Genre"]))
        else:
            print("No books found with the title '{}'.".format(title))
        con.close()

# Function to search books by author
def search_books_by_author(author: str) -> None:
    """
    Searches for books by author.
    """
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select books by author using a wildcard search
        query = "SELECT ID, Title, Author, Genre FROM books WHERE Author LIKE %s"
        cur.execute(query, ('%' + author + '%',))
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["ID", "Title", "Author", "Genre"]))
        else:
            print("No books found by author '{}'.".format(author))
        con.close()

# Function to search books by genre
def search_books_by_genre(genre: str) -> None:
    """
    Searches for books by genre.
    """
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select books by genre using a wildcard search
        query = "SELECT ID, Title, Author, Genre FROM books WHERE Genre LIKE %s"
        cur.execute(query, ('%' + genre + '%',))
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["ID", "Title", "Author", "Genre"]))
        else:
            print("No books found in genre '{}'.".format(genre))
        con.close()

# Function to list available books
def available_books() -> None:
    """
    Lists available books.
    """
    print("\n\t\t\t\tAVAILABLE BOOKS\n")
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select available books where the 'available' column value is 1 (indicating availability)
        query = "SELECT ID, Title, Author, Genre FROM books WHERE available = 1"
        cur.execute(query)
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["ID", "Title", "Author", "Genre"]))
        else:
            print("No books available currently.")
        con.close()

# Function to list issued books
def issued_books() -> None:
    """
    Lists issued books.
    """
    print("\n\t\t\t\tISSUED BOOKS\n")
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select issued books along with their details from the 'issuedbooks' table
        query = "SELECT book_id, issued_book_title, issue_date, due_date, user_id FROM issuedbooks"
        cur.execute(query)
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["Book ID", "Title", "Issue Date", "Due Date", "User ID"]))
        else:
            print("No books issued currently.")
        con.close()

# Function to list overdue books
def overdue_books() -> None:
    """
    Lists overdue books.
    """
    print("\n\t\t\t\tOVERDUE BOOKS\n")
    con = get_database_connection()
    if con:
        cur = con.cursor()
        current_date = datetime.now().strftime('%Y-%m-%d')
        # SQL query to retrieve book details and user emails for books with due dates earlier than the current date (%s)
        query = "SELECT ib.book_id, ib.issued_book_title, ib.due_date, ib.user_id, u.email FROM users AS u INNER JOIN issuedbooks AS ib ON u.id = ib.user_id WHERE ib.due_date < %s"
        cur.execute(query, (current_date,))
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["Book ID", "Title", "Due Date", "User ID", "User Email"]))
        else:
            print("No overdue books.")
        con.close()

# Function to show active users
def active_users() -> None:
    """
    Displays active users who have issued books.
    """
    print("\n\t\t\t\tACTIVE USERS\n")
    con = get_database_connection()
    if con:
        cur = con.cursor()
        # SQL query to select users with an issued book count greater than 0
        query = "SELECT * FROM users WHERE issued_book_count > 0"
        cur.execute(query)
        res = cur.fetchall()
        if res:
            print(tabulate(res, headers=["User ID", "Name", "Email", "Issued Book Count"]))
        else:
            print("No active users found.")
        con.close()

# Function to display the main menu
def menu() -> None:
    """
    Displays the main menu of the Library Management System.
    """
    con = get_database_connection()
    if con:
        print("\n\t\t\t\tLIBRARY MANAGEMENT SYSTEM\n\n")
        print("\t\t\t\t\tMAIN MENU\n")
        print("\t\t1. Add Book\t\t\t 2. Remove Book\n\n\t\t3. Lend Book\t\t\t 4. Return Book\n")
        print("\t\t5. Search Book\t\t\t 6. Available Books\n")
        print("\t\t7. Issued Books\t\t\t 8. Overdue Books\n")
        print("\t\t9. Active Users\t\t\t 10. Exit\n")
        con.close()

# Main function
def main() -> None:
    """
    Main function to run the Library Management System.
    """
    while True:
        menu()
        ch = validate_numeric_input("ENTER YOUR CHOICE:: ")
        if ch == 1:
            add_book()
        elif ch == 2:
            remove_book()
        elif ch == 3:
            lend_book()
        elif ch == 4:
            return_book()
        elif ch == 5:
            search_books()
        elif ch == 6:
            available_books()
        elif ch == 7:
            issued_books()
        elif ch == 8:
            overdue_books()
        elif ch == 9:
            active_users()
        elif ch == 10:
            print("\n\n\t\tTHANK YOU FOR USING THE LIBRARY MANAGEMENT SYSTEM!!\n\n")
            exit()
        else:
            print("PLEASE CHOOSE THE CORRECT CHOICE AND TRY AGAIN!!")

        ch = validate_numeric_input("\n\nPress 0 To Continue Using LMS, Any Other Number To Exit...: ")
        if ch != 0:
            break

if __name__ == "__main__":
    main()
