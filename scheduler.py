import asyncio
from rocketry import Rocketry
from rocketry.conds import every
from database import ConnectionDB

app = Rocketry(config={"task_execution": "async"})


@app.task(every("24 hours", based="finish"))
async def do_mid_night_every_day():
    "This runs for long time"
    try:
        conn = ConnectionDB()
        conn.train_model_audio_history()
        conn = ConnectionDB()
        conn.train_model_user_mental_health_history()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run()
