from datetime import datetime

from pydantic import BaseModel, Field


class RawPositionSchema(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy: int = Field(ge=0)
    timestamp: datetime


class TimelinePathSchema(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime


class VisitSchema(BaseModel):
    start_time: datetime
    end_time: datetime

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    semantic_type: str
    probability: float = Field(ge=0, le=1)
    place_probability: float = Field(ge=0, le=1)


class ActivitySchema(BaseModel):
    start_time: datetime
    end_time: datetime

    start_latitude: float = Field(ge=-90, le=90)
    start_longitude: float = Field(ge=-180, le=180)

    end_latitude: float = Field(ge=-90, le=90)
    end_longitude: float = Field(ge=-180, le=180)

    distance: float = Field(ge=0)

    activity_type: str

    probability: float = Field(ge=0, le=1)
    activity_probability: float = Field(ge=0, le=1)
