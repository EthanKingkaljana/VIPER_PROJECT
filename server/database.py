import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        user="n2nuser",
        password="n2npass",
        database="n2ndb"
    )
