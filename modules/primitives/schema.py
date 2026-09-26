from datetime import datetime

from enum import StrEnum
from typing import Literal
import polars as pl
from pydantic import BaseModel, Field


class PositionSource(StrEnum):
    OBSERVED = "observed"
    MATCHED = "matched"
    INTERPOLATED = "interpolated"


class PositionState(StrEnum):
    STAY = "stay"
    MOVEMENT = "movement"
    REMOVED = "removed"


class RawPositionSchema(BaseModel):
    position_id: int = Field(ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime


class PositionSegmentSchema(BaseModel):
    position_id: int = Field(ge=0)
    segment_id: int = Field(ge=0)
    state: Literal[PositionState.STAY, PositionState.MOVEMENT, PositionState.REMOVED]


class ClusterSchema(BaseModel):
    cluster_id: int = Field(gt=0)

    head_position_id: int = Field(ge=0)
    tail_position_id: int = Field(ge=0)

    point_count: int = Field(gt=0)


class MovementSchema(BaseModel):
    movement_id: int = Field(ge=0)

    head_position_id: int = Field(ge=0)
    tail_position_id: int = Field(ge=0)

    point_count: int = Field(gt=0)


class ProjectedPositionSchema(BaseModel):
    position_id: int = Field(ge=0)
    x: float
    y: float


class CandidatePositionSchema(BaseModel):
    position_id: int = Field(ge=0)

    edge_u: int = Field(ge=0)
    edge_v: int = Field(ge=0)
    edge_key: int = Field(ge=0)

    x: float
    y: float

    distance: float = Field(ge=0)
    distance_along_edge: float = Field(ge=0)


class MatchedPathPointSchema(BaseModel):
    start_position_id: int = Field(ge=0)
    end_position_id: int = Field(ge=0)

    sequence: int = Field(ge=0)

    x: float
    y: float

    path_progress: float = Field(ge=0, le=1)
    timestamp: datetime


class CorrectedPositionSchema(BaseModel):
    position_id: int | None = Field(default=None, ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime


class GeographicPositionSchema(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class CartesianPositionSchema(BaseModel):
    x: float
    y: float


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


def validate_schema_columns(
    df: pl.DataFrame,
    schema: type[BaseModel],
) -> None:
    required_columns = set(schema.model_fields)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")
