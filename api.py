from fastapi import APIRouter, FastAPI
from scheduler import app as app_rocketry
from fastapi.middleware.cors import CORSMiddleware
from database import ConnectionDB
app = FastAPI(
    title="Rocketry with FastAPI",
    description="This is a REST API for a scheduler. It uses FastAPI as the web framework and Rocketry for scheduling."
)
session = app_rocketry.session

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_recommendation = APIRouter(tags=["recommendation"])

@router_recommendation.get("/recommendation")
async def get_recommendation():
    conn = ConnectionDB()
    conn.retrieve_log()
    return "Hello"

app.include_router(router_recommendation)