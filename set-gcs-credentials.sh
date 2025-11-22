#!/bin/bash
# Script to set GCS service account JSON as Cloud Run environment variable
# Usage: ./set-gcs-credentials.sh

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-flash-rock-470212-d5}"
SERVICE_NAME="${SERVICE_NAME:-clippeak-api}"
REGION="${REGION:-europe-west1}"
KEY_FILE="${KEY_FILE:-clippeak-signing-key.json}"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "❌ Error: Key file '$KEY_FILE' not found!"
    echo "   Please create the service account key first:"
    echo "   gcloud iam service-accounts keys create $KEY_FILE \\"
    echo "     --iam-account=clippeak-signing@${PROJECT_ID}.iam.gserviceaccount.com"
    exit 1
fi

# Create a temporary file with properly formatted env vars
ENV_VARS_FILE=$(mktemp)
echo "GCS_SERVICE_ACCOUNT_JSON<<EOF" > "$ENV_VARS_FILE"
cat "$KEY_FILE" >> "$ENV_VARS_FILE"
echo "EOF" >> "$ENV_VARS_FILE"

echo "📝 Setting GCS_SERVICE_ACCOUNT_JSON environment variable..."
echo "   Service: $SERVICE_NAME"
echo "   Region: $REGION"
echo "   Project: $PROJECT_ID"

# Update Cloud Run service with environment variable
gcloud run services update "$SERVICE_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --update-env-vars-file="$ENV_VARS_FILE"

# Clean up
rm "$ENV_VARS_FILE"

echo "✅ Environment variable set successfully!"
echo ""
echo "⚠️  Note: You may need to redeploy your service for the changes to take effect."
echo "   Or wait a few minutes for Cloud Run to pick up the new environment variable."

