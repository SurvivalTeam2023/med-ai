from core.db.db import ConnectionDB


class RecommenderService:
    def get_record_db(self):
        try:
            conn = ConnectionDB()
            users = conn.retrieve_log()
            print(users)
        except Exception as e:
            print(e)
