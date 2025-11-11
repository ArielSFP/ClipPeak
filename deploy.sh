#!/bin/bash
# ClipPeak - Google Cloud Run Deployment Script
# Run this in Google Cloud Shell

set -e  # Exit on any error

# ============================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================
PROJECT_ID="your-project-id"  # CHANGE THIS to your Google Cloud project ID
REGION="europe-west1"
SERVICE_NAME="clippeak-api"
REPO_NAME="clippeak-repo"

echo "============================================================"
echo "ClipPeak - Google Cloud Run Deployment"
echo "============================================================"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Set Project
# ============================================================
echo "📋 Step 1: Setting Google Cloud project..."
gcloud config set project $PROJECT_ID

# ============================================================
# Step 2: Enable Required APIs
# ============================================================
echo ""
echo "🔧 Step 2: Enabling required Google Cloud APIs..."
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
echo "✅ APIs enabled"

# ============================================================
# Step 3: Create Artifact Registry (if not exists)
# ============================================================
echo ""
echo "📦 Step 3: Creating Artifact Registry repository..."
if gcloud artifacts repositories describe $REPO_NAME --location=$REGION &>/dev/null; then
    echo "✅ Repository already exists"
else
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="ClipPeak Docker images"
    echo "✅ Repository created"
fi

# ============================================================
# Step 4: Build with Cloud Build
# ============================================================
echo ""
echo "🏗️  Step 4: Building Docker image with Cloud Build..."
echo "⏱️  This will take 20-30 minutes on first build..."
echo "   (Subsequent builds are faster due to caching)"

IMAGE_URL="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"

gcloud builds submit \
    --tag $IMAGE_URL \
    --timeout=45m \
    --machine-type=e2-highcpu-8

echo "✅ Image built and pushed: $IMAGE_URL"

# ============================================================
# Step 5: Deploy to Cloud Run with GPU
# ============================================================
echo ""
echo "🚀 Step 5: Deploying to Cloud Run with L4 GPU..."

gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_URL \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --memory=16Gi \
    --cpu=8 \
    --timeout=3600 \
    --max-instances=2 \
    --min-instances=0 \
    --concurrency=1 \
    --port=8080 \
    --set-secrets=OPENAI_API_KEY=openai-api-key:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest

echo "✅ Deployment complete!"

# ============================================================
# Step 6: Get Service URL
# ============================================================
echo ""
echo "============================================================"
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "============================================================"

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the health endpoint:"
echo "curl $SERVICE_URL/health"
echo ""
echo "Update your website to use this URL!"
echo "============================================================"

