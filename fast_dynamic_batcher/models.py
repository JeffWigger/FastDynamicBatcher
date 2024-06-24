import typing as t

from pydantic import BaseModel


class Task(BaseModel):
    id: int
    content: t.Any
