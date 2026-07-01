"""FastAPI dependencies for resolving app-state singletons.

We keep all heavy objects (Mongo client, vector store, cross-encoder, the
compiled LangGraph, the chat model) on ``app.state`` so they're built once at
startup and reused for the lifetime of the process. These thin accessors keep
route signatures tidy and make tests easy to override.
"""
from __future__ import annotations

from fastapi import Request

from app.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_mongo(request: Request):
    return request.app.state.mongo


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_grader(request: Request):
    return request.app.state.grader


def get_graph(request: Request):
    return request.app.state.graph


def get_embedder(request: Request):
    return request.app.state.embedder


def get_chat_model(request: Request):
    return request.app.state.chat_model


def get_tracer(request: Request):
    return request.app.state.tracer
