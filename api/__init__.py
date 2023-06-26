from fastapi import APIRouter

from api.auth.auth import auth_router
from api.recommender.recommender import recommender_router

router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["Auth"])
router.include_router(
    recommender_router, prefix="/api/v1/recommendation", tags=["Recommendation"]
)

__all__ = ["router"]
