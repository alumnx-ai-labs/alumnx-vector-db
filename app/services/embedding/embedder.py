from __future__ import annotations

from typing import Iterable

from app.config import get_config


class GeminiEmbedder:
    def __init__(self, model: str | None = None) -> None:
        self.config = get_config()
        self.model = model or self.config.embedding_model
        self.client = self._client()

    def _client(self):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("langchain-google-genai is required for embedding") from exc
        return GoogleGenerativeAIEmbeddings(model=self.model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            vectors.extend(
                self.client.embed_documents(batch)
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)

    def _describe_media(self, file_path: str, mime_type: str | None = None) -> str:
        """Helper method to generate a text description of a media file."""
        import os
        import time
        import mimetypes
        import google.generativeai as genai
        
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        media_file = genai.upload_file(path=file_path, mime_type=mime_type)
        
        # Poll if processing (needed for large files/video) with timeout
        max_wait = 60
        elapsed = 0
        while media_file.state.name == "PROCESSING":
            time.sleep(2)
            elapsed += 2
            media_file = genai.get_file(media_file.name)
            if elapsed > max_wait:
                genai.delete_file(media_file.name)
                raise TimeoutError("Media processing timeout")
            
        if media_file.state.name == "FAILED":
            genai.delete_file(media_file.name)
            raise RuntimeError("Gemini failed to process the media file.")
            
        # Describe the media using a multimodal model
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Describe this media file in extreme detail. "
            "Include visual, structural, and auditory elements. "
            "Transcribe any visible text or audible speech verbatim. "
            "Provide only the description without any introduction."
        )
        
        response = model.generate_content([media_file, prompt])
        if not response.text:
            genai.delete_file(media_file.name)
            raise RuntimeError("Empty response from Gemini")
            
        description = response.text
        
        # Cleanup remote file
        genai.delete_file(media_file.name)
        
        return description

    def embed_image(self, image_path: str, mime_type: str | None = None) -> dict[str, any]:
        """Generates an embedding for an image by describing it first."""
        description = self._describe_media(image_path, mime_type)
        return {
            "description": description,
            "embedding": self.embed_query(description)
        }

    def embed_audio(self, audio_path: str, mime_type: str | None = None) -> dict[str, any]:
        """Generates an embedding for an audio file by describing/transcribing it first."""
        description = self._describe_media(audio_path, mime_type)
        return {
            "description": description,
            "embedding": self.embed_query(description)
        }

    def embed_video(self, video_path: str, mime_type: str | None = None) -> dict[str, any]:
        """Generates an embedding for a video file by describing it first."""
        description = self._describe_media(video_path, mime_type)
        return {
            "description": description,
            "embedding": self.embed_query(description)
        }

    def embed_text_file(self, text_path: str) -> dict[str, any]:
        """Generates an embedding for a text file by reading its contents directly."""
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "description": content,
            "embedding": self.embed_query(content)
        }
