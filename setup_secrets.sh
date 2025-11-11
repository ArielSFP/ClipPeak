#!/bin/bash
# ClipPeak - Setup Google Cloud Secrets
# Run this ONCE before deploying

set -e

PROJECT_ID="your-project-id"  # CHANGE THIS

echo "============================================================"
echo "ClipPeak - Setting up Google Cloud Secrets"
echo "============================================================"
echo ""

gcloud config set project $PROJECT_ID

# ============================================================
# Create Secrets (you'll be prompted to enter values)
# ============================================================

echo "📝 Creating secrets for API keys..."
echo ""
echo "⚠️  IMPORTANT: You'll be prompted to enter each API key."
echo "   Press Ctrl+D after pasting each key."
echo ""

# OpenAI API Key
echo "1️⃣  Enter your OpenAI API Key (starts with sk-proj-...):"
gcloud secrets create openai-api-key --data-file=-
echo "✅ OpenAI API key saved"
echo ""

# Supabase URL
echo "2️⃣  Enter your Supabase URL (https://...supabase.co/):"
gcloud secrets create supabase-url --data-file=-
echo "✅ Supabase URL saved"
echo ""

# Supabase Key
echo "3️⃣  Enter your Supabase Anon Key (starts with eyJhbGci...):"
gcloud secrets create supabase-key --data-file=-
echo "✅ Supabase key saved"
echo ""

# ============================================================
# Grant Access to Cloud Run Service Account
# ============================================================

echo "🔐 Granting Cloud Run access to secrets..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding openai-api-key \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding supabase-url \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding supabase-key \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role=roles/secretmanager.secretAccessor

echo "✅ Secrets configured successfully!"
echo ""
echo "============================================================"
echo "Next step: Run ./deploy.sh to deploy the service"
echo "============================================================"

