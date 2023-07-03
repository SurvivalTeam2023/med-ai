import asyncio
from rocketry import Rocketry
from rocketry.conds import every
from database import ConnectionDB
app = Rocketry(config={"task_execution": "async"})

@app.task(every('20 seconds', based="finish"))
async def do_mid_night_every_day():
    "This runs for long time"
    try:
        conn = ConnectionDB()
        conn.retrieve_log()
    except Exception as e:
            print(e)

if __name__ == "__main__":
    app.run()