from datetime import datetime

import polars as pl
from pydantic import BaseModel, Field


class RawPositionSchema(BaseModel):
    position_id: int = Field(ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime


class PositionSegmentSchema(BaseModel):
    position_id: int = Field(ge=0)
    segment_id: int = Field(ge=0)


class SegmentSchema(BaseModel):
    segment_id: int = Field(ge=0)

    mean_latitude: float = Field(ge=-90, le=90)
    mean_longitude: float = Field(ge=-180, le=180)

    head_position_id: int = Field(ge=0)
    tail_position_id: int = Field(ge=0)

    point_count: int = Field(gt=0)

    prev_next_distance: float | None = Field(default=None, ge=0)


class PositionClusterSchema(BaseModel):
    position_id: int = Field(ge=0)
    cluster_id: int = Field(ge=0)


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
