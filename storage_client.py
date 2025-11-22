"""
Storage Client Abstraction Layer
Supports Google Cloud Storage (primary) with fallback interface for Supabase
"""
from google.cloud import storage
import os
from typing import Optional, Union
import io

class StorageClient:
    """Abstraction layer for storage operations - GCS implementation."""
    
    def __init__(self):
        # GCS bucket names (hardcoded)
        self.videos_bucket_name = "clippeak-videos"
        self.processed_bucket_name = "clippeak-processed-videos"
        
        # Initialize GCS client (uses default credentials or GOOGLE_APPLICATION_CREDENTIALS)
        self.client = storage.Client()
        self.videos_bucket = self.client.bucket(self.videos_bucket_name)
        self.processed_bucket = self.client.bucket(self.processed_bucket_name)
    
    def download(self, bucket_name: str, file_key: str) -> bytes:
        """
        Download file from storage.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Path to file in bucket (e.g., "user_id/video_folder/file.mp4")
        
        Returns:
            bytes: File content
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {file_key} in bucket {bucket_name}")
        
        return blob.download_as_bytes()
    
    def upload_from_file(self, bucket_name: str, file_key: str, file_path: str, content_type: Optional[str] = None) -> str:
        """
        Upload file from local filesystem to storage.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Destination path in bucket
            file_path: Local file path
            content_type: MIME type (optional)
        
        Returns:
            str: File key (path in bucket)
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        
        if content_type:
            blob.content_type = content_type
        
        blob.upload_from_filename(file_path)
        return file_key
    
    def upload_from_bytes(self, bucket_name: str, file_key: str, data: bytes, content_type: Optional[str] = None) -> str:
        """
        Upload bytes to storage.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Destination path in bucket
            data: File content as bytes
            content_type: MIME type (optional)
        
        Returns:
            str: File key (path in bucket)
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        
        if content_type:
            blob.content_type = content_type
        
        blob.upload_from_string(data, content_type=content_type)
        return file_key
    
    def delete(self, bucket_name: str, file_keys: Union[str, list[str]]):
        """
        Delete files from storage.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_keys: Single file key (str) or list of file keys
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        
        # Handle single file or list
        if isinstance(file_keys, str):
            file_keys = [file_keys]
        
        blobs = [bucket.blob(key) for key in file_keys]
        bucket.delete_blobs(blobs, on_error=lambda blob, error: print(f"Failed to delete {blob.name}: {error}"))
    
    def get_public_url(self, bucket_name: str, file_key: str) -> str:
        """
        Get public URL for a file (if bucket is public).
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Path to file in bucket
        
        Returns:
            str: Public URL
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        return blob.public_url
    
    def create_signed_url(self, bucket_name: str, file_key: str, expiration_seconds: int = 3600, method: str = "GET", content_type: str = None) -> str:
        """
        Create a signed URL for temporary access.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Path to file in bucket
            expiration_seconds: URL expiration time in seconds (default: 1 hour)
            method: HTTP method ("GET" for downloads, "PUT" for uploads)
            content_type: Content type for PUT uploads (required for browser uploads)
        
        Returns:
            str: Signed URL
        """
        from datetime import timedelta
        
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        
        # For PUT uploads, we need to specify content-type in the signature
        # This ensures the browser can upload with the correct content-type header
        if method == "PUT" and content_type:
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration_seconds),
                method=method,
                content_type=content_type
            )
        else:
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration_seconds),
                method=method
            )
    
    def exists(self, bucket_name: str, file_key: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            file_key: Path to file in bucket
        
        Returns:
            bool: True if file exists
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blob = bucket.blob(file_key)
        return blob.exists()
    
    def list_files(self, bucket_name: str, prefix: str = "") -> list[str]:
        """
        List files in a bucket with optional prefix filter.
        
        Args:
            bucket_name: "videos" or "processed-videos"
            prefix: Prefix to filter files (e.g., "user_id/video_folder/")
        
        Returns:
            list[str]: List of file keys (paths)
        """
        bucket = self.videos_bucket if bucket_name == "videos" else self.processed_bucket
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

# Global instance (singleton pattern)
_storage_client: Optional[StorageClient] = None

def get_storage_client() -> StorageClient:
    """Get or create the global storage client instance."""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client

# Convenience alias
storage_client = get_storage_client()

