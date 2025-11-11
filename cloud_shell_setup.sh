#!/bin/bash
# ClipPeak - One-Click Setup for Google Cloud Shell
# This script sets up everything you need to deploy ClipPeak to Cloud Run

set -e  # Exit on error

echo "============================================================"
echo "ClipPeak - Cloud Run Setup Wizard"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Get Project ID
# ============================================================
echo "Step 1: Project Configuration"
echo "------------------------------"

# Try to detect current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")

if [ -z "$CURRENT_PROJECT" ]; then
    echo "⚠️  No project currently set"
    echo "📋 Available projects:"
    gcloud projects list --format="table(projectId,name)"
    echo ""
    read -p "Enter your Project ID: " PROJECT_ID
else
    echo "Current project: $CURRENT_PROJECT"
    read -p "Use this project? (y/n): " USE_CURRENT
    if [ "$USE_CURRENT" = "y" ] || [ "$USE_CURRENT" = "Y" ]; then
        PROJECT_ID=$CURRENT_PROJECT
    else
        read -p "Enter your Project ID: " PROJECT_ID
    fi
fi

gcloud config set project $PROJECT_ID
echo "✅ Using project: $PROJECT_ID"
echo ""

# ============================================================
# STEP 2: Configuration
# ============================================================
REGION="europe-west1"
SERVICE_NAME="clippeak-api"
REPO_NAME="clippeak-repo"
IMAGE_URL="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"

echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service Name: $SERVICE_NAME"
echo ""

# ============================================================
# STEP 3: Enable APIs
# ============================================================
echo "Step 2: Enabling Google Cloud APIs..."
echo "------------------------------"
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
echo "✅ APIs enabled"
echo ""

# ============================================================
# STEP 4: Create Artifact Registry
# ============================================================
echo "Step 3: Setting up Artifact Registry..."
echo "------------------------------"
if gcloud artifacts repositories describe $REPO_NAME --location=$REGION &>/dev/null; then
    echo "✅ Repository already exists"
else
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="ClipPeak Docker images"
    echo "✅ Repository created"
fi
echo ""

# ============================================================
# STEP 5: Setup Secrets (API Keys)
# ============================================================
echo "Step 4: Setting up API Keys as Secrets..."
echo "------------------------------"
echo "⚠️  You'll need to enter your API keys"
echo ""

# Check if secrets already exist
setup_secret() {
    SECRET_NAME=$1
    SECRET_DESCRIPTION=$2
    
    if gcloud secrets describe $SECRET_NAME &>/dev/null; then
        echo "✅ Secret '$SECRET_NAME' already exists"
        read -p "   Update it? (y/n): " UPDATE_SECRET
        if [ "$UPDATE_SECRET" = "y" ] || [ "$UPDATE_SECRET" = "Y" ]; then
            echo "   Enter new value for $SECRET_DESCRIPTION:"
            echo "   (Type or paste, then press Ctrl+D when done)"
            gcloud secrets versions add $SECRET_NAME --data-file=-
            echo "✅ Secret updated"
        fi
    else
        echo "Creating secret: $SECRET_NAME"
        echo "Enter $SECRET_DESCRIPTION:"
        echo "(Type or paste, then press Ctrl+D when done)"
        gcloud secrets create $SECRET_NAME --data-file=-
        echo "✅ Secret created"
    fi
}

setup_secret "openai-api-key" "OpenAI API Key (sk-proj-...)"
setup_secret "supabase-url" "Supabase URL (https://...supabase.co/)"
setup_secret "supabase-key" "Supabase Anon Key (eyJhbGci...)"

echo ""
echo "🔐 Granting Cloud Run access to secrets..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

for SECRET in openai-api-key supabase-url supabase-key; do
    gcloud secrets add-iam-policy-binding $SECRET \
        --member=serviceAccount:$SERVICE_ACCOUNT \
        --role=roles/secretmanager.secretAccessor \
        --quiet
done

echo "✅ Secrets configured"
echo ""

# ============================================================
# STEP 6: Build Docker Image
# ============================================================
echo "Step 5: Building Docker Image..."
echo "------------------------------"
echo "⏱️  This will take 20-30 minutes (first time only)"
echo "   Subsequent builds are much faster due to caching"
echo ""
read -p "Start build now? (y/n): " START_BUILD

if [ "$START_BUILD" = "y" ] || [ "$START_BUILD" = "Y" ]; then
    gcloud builds submit \
        --tag $IMAGE_URL \
        --timeout=45m \
        --machine-type=e2-highcpu-8
    
    echo "✅ Image built successfully!"
else
    echo "⏭️  Skipping build. Run this command manually:"
    echo "   gcloud builds submit --tag $IMAGE_URL --timeout=45m"
    exit 0
fi
echo ""

# ============================================================
# STEP 7: Deploy to Cloud Run
# ============================================================
echo "Step 6: Deploying to Cloud Run with L4 GPU..."
echo "------------------------------"

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
echo ""

# ============================================================
# STEP 8: Get Service URL and Test
# ============================================================
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "============================================================"
echo "🎉 SUCCESS! ClipPeak is now running on Cloud Run!"
echo "============================================================"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "🧪 Testing health endpoint..."
curl -s $SERVICE_URL/health | python3 -m json.tool
echo ""
echo "============================================================"
echo "📝 Next Steps:"
echo "============================================================"
echo "1. Update your website to use: $SERVICE_URL"
echo "2. Test by uploading a video from clippeak.co.il"
echo "3. Monitor costs at: console.cloud.google.com/billing"
echo "4. View logs: gcloud run services logs tail $SERVICE_NAME --region=$REGION"
echo ""
echo "💰 Cost: ~\$0/hour when idle, ~\$1.20/hour when processing"
echo "   (Only charged per second while container runs!)"
echo "============================================================"

