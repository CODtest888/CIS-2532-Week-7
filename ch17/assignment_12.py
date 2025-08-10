#########################################################################
#Application Name: assignment_12.py
#Description:      Examples from chapter 17 showing how to run database queries
#Author:           Paul Dietel, Harvey Deitel / Intor to Python for Computer Science and Data Science
#Comments:         Ryan Buechler
#Course Name:      CIS-2532
#Section:          NET01
#Instructor:       Mohammad Morovati
#Assignment#:      Assignment #12
#Date:             08/09/2025
#########################################################################


import sqlite3

# Sets a variable for connecting to the books database
connection = sqlite3.connect('books.db')

import pandas as pd

pd.options.display.max_columns = 10

# Stores the data from the authors table using the authors id as the index value
df = pd.read_sql('SELECT * FROM authors', connection, index_col=['id'])

print(df)
print()

# Stores the data from the titles table
df = pd.read_sql('SELECT * FROM titles', connection)

print(df)
print()

# Stores the data from the author_ISBN table
df = pd.read_sql('SELECT * FROM author_ISBN', connection)

print(df.head())
print()

# Stores only the first and last name columns from the authors table
df = pd.read_sql('SELECT first, last FROM authors', connection)

print(df)
print()

# Stores three columns from the titles table and only titles copyrighted after 2016
df = pd.read_sql("""SELECT title, edition, copyright FROM titles WHERE copyright > '2016'""", connection)

print(df)
print()

# Stores the authors whose last names start with a D
df = pd.read_sql("""SELECT id, first, last FROM authors WHERE last LIKE 'D%'""", connection, index_col=['id'])

print(df)
print()

# Stores any author whose name has a b in it that is proceeded by a character
df = pd.read_sql("""SELECT id, first, last FROM authors WHERE first LIKE '_b%'""", connection, index_col=['id'])

print(df)
print()

# Stores titles table in ascending order according to the title column
df = pd.read_sql('SELECT title FROM titles ORDER BY TITLE ASC', connection)

print(df)
print()

# Stores the authors names sorted in ascending order by last name and then by first name 
df = pd.read_sql('SELECT id, first, last FROM authors ORDER BY last, first', connection, index_col=['id'])

print(df)
print()

# Stores the authors names sorted in descending order by last name and then by ascending order by first name
df = pd.read_sql('SELECT id, first, last FROM authors ORDER BY last DESC, first ASC', connection, index_col=['id'])

print(df)
print()

# Stores books information with titles that end with 'How to Program' and sorts them in ascending order
df = pd.read_sql("""SELECT isbn, title, edition, copyright
                    FROM titles
                    WHERE title LIKE '%How to Program'
                    ORDER BY title""", connection)

print(df)
print()

# Stores the first five rows of a merged table of the authors table and author_ISN table
# The author id is used to match the isbn to the names
df = pd.read_sql("""SELECT first, last, isbn
                    FROM authors
                    INNER JOIN author_ISBN
                    ON authors.id = author_ISBN.id
                    ORDER BY last, first""", connection).head()

print(df)
print()

# Creates a cursor object to use in the database
cursor = connection.cursor()

# Uses the cursor to add a new row to the author table
cursor = cursor.execute("""INSERT INTO authors (first, last)
                          Values ('Sue', 'Red')""")

# Stores the author table with the additional row
df = pd.read_sql('SELECT id, first, last FROM Authors', connection, index_col=['id'])

print(df)
print()

# Changes the last name of the author with the name Sue Red
cursor = cursor.execute("""UPDATE authors SET last='Black'
                           WHERE last='Red' AND first = 'Sue'""")

# Displays how many rows where changed
print(cursor.rowcount)

df = pd.read_sql('SELECT id, first, last FROM authors', connection, index_col=['id'])

print(df)
print()

# Deletes the row with the unique id of 6
cursor = cursor.execute("""DELETE FROM authors
                           WHERE id=6""")

print(cursor.rowcount)

# Stores the author table after deletion of a row
df = pd.read_sql('SELECT id, first, last FROM authors', connection, index_col=['id'])

print(df)
print()
