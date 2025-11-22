"""
Storage Client Abstraction Layer
Supports Google Cloud Storage (primary) with fallback interface for Supabase
"""
from google.cloud import storage
from google.oauth2 import service_account
import os
from typing import Optional, Union
import io
import json

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
    
    def _get_signing_credentials(self):
        """
        Get credentials with private key for signing URLs.
        Cloud Run's default credentials don't have a private key, so we need
        a service account JSON key for signing.
        
        Returns:
            Service account credentials with private key, or None to use default
        """
        # ⚠️ SECURITY WARNING: Hardcoded service account credentials
        # In production, use environment variables or Secret Manager instead!
        # Try to load from local JSON file first (for development)
        local_key_files = [
            "clippeak-signing-key.json",
            "/app/clippeak-signing-key.json",  # Cloud Run path
            os.path.join(os.path.dirname(__file__), "clippeak-signing-key.json")
        ]
        
        for key_file in local_key_files:
            if os.path.exists(key_file):
                try:
                    with open(key_file, 'r') as f:
                        service_account_info = json.load(f)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info
                    )
                    print(f"✅ Using service account credentials from file {key_file} for signing: {credentials.service_account_email}")
                    return credentials
                except Exception as e:
                    print(f"⚠️  Failed to load from {key_file}: {e}")
                    continue
        
        # Hardcoded fallback (⚠️ INSECURE - for convenience only)
        # NOTE: You need to paste the COMPLETE private_key from your JSON file here
        # The private key should start with "-----BEGIN PRIVATE KEY-----" and end with "-----END PRIVATE KEY-----"
        # Replace the placeholder below with your complete private key
        HARDCODED_SERVICE_ACCOUNT = {
            "type": "service_account",
            "project_id": "flash-rock-470212-d5",
            "private_key_id": "79a72ca162e16b09b3fd0d8dcf39fb58b66cc570",
            # ⚠️ REPLACE THIS with the complete private_key from your clippeak-signing-key.json file
            # Copy the ENTIRE value from the "private_key" field - it's a very long string
            # It should include ALL lines between BEGIN and END markers
            # IMPORTANT: Keep the \n characters for newlines - just paste the key as-is from JSON
            "private_key": "",  # Paste your complete private_key here
            "client_email": "clippeak-signing@flash-rock-470212-d5.iam.gserviceaccount.com",
            "client_id": "111976950819230532180",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/clippeak-signing%40flash-rock-470212-d5.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }
        
        # Priority 1: Check for service account JSON key in environment variable (Cloud Run secret)
        service_account_json = os.environ.get("GCS_SERVICE_ACCOUNT_JSON")
        
        if service_account_json:
            try:
                # Parse JSON string from environment variable
                service_account_info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info
                )
                print(f"✅ Using service account credentials from env var for signing: {credentials.service_account_email}")
                return credentials
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️  Failed to load service account JSON from GCS_SERVICE_ACCOUNT_JSON: {e}")
                print("   Falling back to hardcoded credentials")
        
        # Priority 2: Check for GOOGLE_APPLICATION_CREDENTIALS (path to JSON file)
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                print(f"✅ Using service account credentials from file: {credentials_path}")
                return credentials
            except Exception as e:
                print(f"⚠️  Failed to load credentials from {credentials_path}: {e}")
                print("   Falling back to hardcoded credentials")
        
        # Priority 3: Use hardcoded service account (⚠️ SECURITY RISK - for convenience only)
        try:
            credentials = service_account.Credentials.from_service_account_info(
                HARDCODED_SERVICE_ACCOUNT
            )
            print(f"✅ Using hardcoded service account credentials for signing: {credentials.service_account_email}")
            print("⚠️  WARNING: Using hardcoded credentials! Consider using environment variables or Secret Manager.")
            return credentials
        except Exception as e:
            print(f"❌ Failed to load hardcoded service account credentials: {e}")
            print("   Signed URL generation will fail.")
            return None
    
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
        
        # Get credentials with private key for signing
        signing_credentials = self._get_signing_credentials()
        
        # Generate signed URL with credentials (if available)
        if signing_credentials:
            # Create a new blob with the signing credentials
            signing_client = storage.Client(credentials=signing_credentials)
            signing_bucket = signing_client.bucket(bucket.name)
            signing_blob = signing_bucket.blob(file_key)
            
            # For PUT uploads, we need to specify content-type in the signature
            if method == "PUT" and content_type:
                return signing_blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration_seconds),
                    method=method,
                    content_type=content_type
                )
            else:
                return signing_blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration_seconds),
                    method=method
                )
        else:
            # Fallback to default (will fail but provides better error message)
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

