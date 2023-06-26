from fastapi import APIRouter, Response

from app.auth.services.jwt import JwtService
from api.recommender.request.recommender import RequestCommenderRequest
from api.recommender.response.recommender import RecommenderResponse
from app.recommender.services.main import RecommenderService

recommender_router = APIRouter()


@recommender_router.post(
    "/get_recommender",
    response_model=RecommenderResponse,
    responses={"400": {"model": {}}},
)
async def refresh_token(request: RequestCommenderRequest):
    recommender_service = RecommenderService()
    recommender_service.get_record_db()
    return {"response": "hello"}
