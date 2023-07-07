from typing import Annotated

from fastapi import Depends
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseSettings
from supabase.client import Client, create_client
from vectorstore.supabase import SupabaseVectorStore


class BrainRateLimiting(BaseSettings):
    max_brain_size: int = 52428800
    max_brain_per_user: int = 5


class BrainSettings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    supabase_url: str
    supabase_service_key: str
    resend_api_key: str = "null"
    resend_email_address: str = "brain@mail.quivr.app"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_type: str = "open_ai"
    openai_gpt_deployment_id : str = None   # pyright: ignore reportPrivateUsage=none
    openai_embedding_deployment_id : str = None  # pyright: ignore reportPrivateUsage=none


class LLMSettings(BaseSettings):
    private: bool = False
    model_path: str = "./local_models/ggml-gpt4all-j-v1.3-groovy.bin"


def common_dependencies() -> dict:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        deployment=settings.openai_embedding_deployment_id,
        openai_api_type=settings.openai_api_type,
    )
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    documents_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="vectors"
    )
    summaries_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="summaries"
    )

    return {
        "supabase": supabase_client,
        "embeddings": embeddings,
        "documents_vector_store": documents_vector_store,
        "summaries_vector_store": summaries_vector_store,
    }


CommonsDep = Annotated[dict, Depends(common_dependencies)]
