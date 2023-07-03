from sqlalchemy import create_engine, text
import datetime
import pandas as pd
import os
class ConnectionDB:
    def __init__(self):
        engine = create_engine(
             f'mysql+pymysql://{os.environ.get("DB_USER", "root")}:{os.environ.get("DB_PASSWORD", "JCpulWbTp5")}@{os.environ.get("DB_HOST", "14.225.204.74")}:3306/{os.environ.get("DB_DATABASE", "med-dump-ai")}'
        )
        self.connection = engine.connect()
        print("db_connected")

    def close(self):
        if hasattr(self, 'connection') and self.connection is not None:
            self.connection.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Exception occurred while closing connection: {e}")
    def create_path_file(self, prefix):
        current_path = os.getcwd()
        folder_path = os.path.join(current_path, "train_model")
        if not os.path.exists(folder_path):
            os.umask(0)
            os.makedirs(folder_path,mode=0o777, exist_ok=False)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return os.path.join(folder_path,  prefix + timestamp + ".txt")
    
    def train_model_audio_history(self):
        prefix = "train_model_music_history"
        filename = self.create_path_file(prefix)
        try:
            query = text(
                """
                SELECT h.audio_id AS audio_id, h.user_id, a.name AS audio_name, a2.username AS artist_name, h.count
                FROM history h
                INNER JOIN audio a ON a.id = h.audio_id
                INNER JOIN artist a2 ON a2.id = a.artist_id
                """
                )
            df = pd.read_sql(query, self.connection)
            output_file = filename
            df.to_csv(output_file, sep="\t", index=False)
            self.close()  # Close the connection before returning the result
            return None
        except Exception as e:
            print(e)
             
    def train_model_user_genre_history(self):
        prefix = "train_model_genre_history"
        filename = self.create_path_file(prefix)
        try:
            query = text(
                """
              
                """
                )
            df = pd.read_sql(query, self.connection)
            output_file = filename
            df.to_csv(output_file, sep="\t", index=False)
            self.close()  # Close the connection before returning the result
            return None
        except Exception as e:
            print(e)