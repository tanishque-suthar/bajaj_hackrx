from pydantic import BaseModel, HttpUrl, Field
from typing import List

class HackRXRequest(BaseModel):
    """Request model for /hackrx/run endpoint"""
    documents: str = Field(
        ...,
        description="Blob URL of the document to process",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03..."
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of questions to answer based on the document",
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?"
                ]
            }
        }

class HackRXResponse(BaseModel):
    """Response model for /hackrx/run endpoint"""
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions",
        example=[
            "A grace period of thirty days is provided for premium payment after the due date...",
            "There is a waiting period of thirty-six (36) months of continuous coverage..."
        ]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
                ]
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., example="healthy")
    message: str = Field(..., example="HackRX API is running")
    version: str = Field(..., example="1.0.0")
