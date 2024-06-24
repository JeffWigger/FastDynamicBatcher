import typing as t

from pydantic import BaseModel


class Task(BaseModel):
    """A model of a task used by the DynBatcher. It is used to return the batched inputs back to the right calling thread."""

    id: int
    content: t.Any
