#########################################################################
#Application Name: assignment_12_ex17-1.py
#Description:      Creates three training models to predict house pricing
#Comments:         Ryan Buechler
#Course Name:      CIS-2532
#Section:          NET01
#Instructor:       Mohammad Morovati
#Assignment#:      Assignment #12
#Date:             08/09/2025
#########################################################################


import sqlite3
import pandas as pd

# Creates a variable that connects to the books database
connection = sqlite3.connect('books.db')

# Stores the last names of the authors in descending order
df = pd.read_sql('SELECT last FROM authors ORDER BY last DESC', connection)

print('Part A\n')
print(df)
print()

# Stores the book titles in ascending order
df = pd.read_sql('SELECT title FROM titles ORDER BY TITLE ASC', connection,)

print('Part B\n')
print(df)
print()

# Stores the isbn, book title, and copyright year from all the books written by
# one author in ascending order by book title
df = pd.read_sql("""SELECT author_ISBN.isbn, title, copyright
                    FROM author_ISBN
                    INNER JOIN authors
                    ON author_ISBN.id = authors.id
                    INNER JOIN titles
                    ON author_ISBN.isbn = titles.isbn
                    WHERE author_ISBN.id = 2
                    ORDER BY title ASC""", connection)

print('Part C\n')
print(df)
print()

# Creates a cursor object to manipulate table rows
cursor = connection.cursor()

# Adds a new author to the author table
cursor = cursor.execute("""INSERT INTO authors (first, last)
                           Values ('Ray', 'Romano')""")

# Stores the author table
df = pd.read_sql("""SELECT * FROM authors""", connection)

print('Part D\n')
print(df)
print()

# Adds a new book written by the new author to the titles table and author_ISBN table
cursor = cursor.execute("""INSERT INTO titles (isbn, title, edition, copyright)
                           Values ('0123456789', 'Everybody Loves Raymond', '1', '2025')""")

cursor = cursor.execute("""INSERT INTO author_ISBN (id, isbn)
                           Values ('6', '0123456789')""")

# Stores the titles table
df = pd.read_sql("""SELECT * FROM titles""", connection)

print('Part D\n')
print(df)
print()

# Stores the author_ISBN table
df = pd.read_sql("""SELECT * FROM author_ISBN""", connection)

print(df)
