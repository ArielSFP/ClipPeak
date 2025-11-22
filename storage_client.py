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
            # Hardcoded private key from clippeak-signing-key.json
            # ⚠️ SECURITY: This is hardcoded for convenience - consider using env var or Secret Manager in production
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCiyNn8XqQXpze0\nXFYEssbhQXqSE6aAGGF4FHyw9XaAU417o16CnHz0xvVnwf9WV3ARPLNrg5fG/uMa\nkYMKpjJV9r6BF9bGkxJmoRTLCdaU9f1ovRR5FAriRWyJHCdXCagUtUbGm7/Y1vMs\nsf00UFgm7fDYgdJKBO3b8gOWEnFKfXcgWgvCgf0HGy8udfcCZQAV+dra3s3KPJiW\nBd8vntVRTRbv4yq6Hw1l5sJmluWIMWL9zBOUOWoBO6hE0OTChcKz9vMN0RNQ1gpc\nm1asy4n/RWfYSQmH5OBbROtZliobmsW6tHbP073y87I5AOPDGnzE6o7Iwun5Vw/8\nmYYhbz0pAgMBAAECggEARTJQ62UFpuJXnQ0lKZEfVnCBnGfK/qeegy9eQ9DMA0fB\nHD35dlb0DQ8oEWeXVUoM4t7lO/4vvhDqVAhn5lZ4ji85kuGh1D5/c23ky1xOwXqB\nsoyxj7hz2fR96xizJl39+3OFdvaNCi1xKF4uzpaaqk259CbXT8yhmb3aRWhojStz\npgHHbM8x734dkeSugotg0pokiECG+PHx8AHjx46kgOK+SOmDWM/4+Bq54lBxBpvB\nBxABtITYFMa2NkIJvmQa027vjMCqix1rOnQvxmIwHfvS4IHe3ZkD0JXxTHr7WtDV\nzVbTo++Dmjx5hmTtvrSL/65QzwEmWzvIFF+9McLbUwKBgQDRG1vwMxNxLmuMsGR+\nnHrj4PqhdJf9QnqwJO9pgd34Njwhn6cTH6VnfbNq9ZWgRCh2gkAESLABjIaxYQDA\n1Lo7h0wjEhsENFHCeCrl7Pkl8YVlU3srAmqim0qD/SKikDZbMFLcVxfEDNkP1mAu\njC1b6z80RMp4GDpCw+9fFzFOJwKBgQDHSiuYZWPCqKq8e/rewEFntgdeif2CbkRg\n8QEbEEXElqMUjQ5SkXGa16h0ZFWjkzY8JEarPeUQ1ItBGllmncMi3k2Yq9AW7vmU\nO8xMkzLu/omm3vG9paCaTIoy7FInzGn5gF0ehPa8nPYMoOZCp6vyX5F7MJBxETG5\nCR7z0aF8LwKBgQDF5rfPj/dhkcZV44do0J351648agWCuo008e/oEfHK/UFRrTDJ\nh5TaanwIyZ6vjHIjHsiJhWWWQbIkZCL7XRlYR4dGVddd1kpe2631554xS/Cs4iJa\npTHY7VHLsyI5anUXPpfdU0v5zxMLyx9CToaLTQyHYmeg24UgNSHyBBg01wKBgDOi\nT/FIM7S3pqT3KmSUQ13vl7Ey52/nyeekCblNWgY5UT993HwgGuMXhRnC9T1QYvjq\nHTU48CSQLRRNE2S281wRZOFtgpiuihOR+rjMWqH5aQjHMrMEOrw/oH7ZI+AT4lE2\npOnHYXY05LsY5Q/YCYDeqf7U8leZMNzUNCTP55sZAoGAGY9J1PHjKv0dm7AX0wl0\n9lzzxGMA6ar7OUtNSIcbg3MwYeEVtTnvaKmFZgPnVgjhwmHLWUgcX3qzek1iV7+x\nVHD+D5T/HhFccXdYwsNRd8lCkhKzPsJ2NO+ILeZRrYHUBRLUSpxcqnAz8OfINvTC\nCu8ZhM+jbXB4bsnD9TwjzJY=\n-----END PRIVATE KEY-----\n",
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

