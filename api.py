from fastapi import APIRouter, FastAPI
from scheduler import app as app_rocketry
from fastapi.middleware.cors import CORSMiddleware
from database import ConnectionDB
from services import get_audio_ids_recommend_by_user_id, get_audio_similar_with_song_id

app = FastAPI(
    title="Rocketry with FastAPI",
    description="This is a REST API for a scheduler. It uses FastAPI as the web framework and Rocketry for scheduling.",
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


@router_recommendation.get("/recommendation/user/")
async def get_recommendation_by_user_id(user_id):
    return get_audio_ids_recommend_by_user_id(user_id=user_id)


@router_recommendation.get("/recommendation/audio/")
async def get_recommendation_by_audio_id(audio_id):
    return get_audio_similar_with_song_id(audio_id=audio_id)


@router_recommendation.get("/force_train_model")
async def force_train_model():
    conn = ConnectionDB()
    conn.train_model_audio_history()
    return True


app.include_router(router_recommendation)
