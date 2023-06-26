from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import config
from sqlalchemy.engine.url import URL

url = URL(
    drivername="mysql+pymysql",
    username=config.DB_USER,
    password=config.DB_PASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT,
    database=config.DB_DATABASE,
)

# engine = create_engine(url)

# query = "SELECT * FROM users"
# result = engine.execute(query)
# for row in result:
#     # Access the values in each row
#     id = row["id"]
#     name = row["username"]
#     email = row["email"]

#     # Process the values as needed
#     print(id, name, email)

# # Close the result set
# result.close()
