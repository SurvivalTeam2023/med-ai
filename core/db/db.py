from core.config import config
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL


class ConnectionDB:
    def __init__(self):
        engine = create_engine(
            URL(
                drivername="mysql+pymysql",
                username=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT,
                database=config.DB_DATABASE,
            )
        )
        self.connection = engine.connect()

    def __del__(self):
        self.connection.close()

    def retrieve_log(self):
        query = text("SELECT * FROM history")
        result = self.connection.execute(query)
        if result.rowcount != 0:
            return result.first()
        return None
