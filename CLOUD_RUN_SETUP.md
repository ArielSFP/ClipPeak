# ClipPeak - Google Cloud Run Setup Guide

## Complete Setup from Google Cloud Shell (No Local Installation Needed!)

### Prerequisites
- Google Cloud account
- GitHub repo: https://github.com/ArielSFP/ClipPeak
- L4 GPU quota in europe-west1 region

---

## 🚀 Step-by-Step Setup (Cloud Shell)

### Step 1: Open Cloud Shell
1. Go to https://console.cloud.google.com
2. Click the **terminal icon** (>_) in top right corner
3. Cloud Shell opens (you get a free Linux terminal with 5GB storage!)

---

### Step 2: Initial Configuration

```bash
# Set your project ID (find it in console.cloud.google.com)
export PROJECT_ID="your-project-id-here"  # CHANGE THIS!

# Set project as default
gcloud config set project $PROJECT_ID

# Verify it's set
gcloud config get-value project
```

---

### Step 3: Enable Required APIs

```bash
# Enable all needed Google Cloud APIs (takes ~2 minutes)
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com

echo "✅ APIs enabled"
```

---

### Step 4: Clone Your Repository

```bash
# Clone your GitHub repo
git clone https://github.com/ArielSFP/ClipPeak.git
cd ClipPeak

# Verify files are there
ls -la
```

---

### Step 5: Create Artifact Registry Repository

```bash
# Create Docker image repository
gcloud artifacts repositories create clippeak-repo \
    --repository-format=docker \
    --location=europe-west1 \
    --description="ClipPeak Docker images"

echo "✅ Artifact Registry created"
```

---

### Step 6: Set Up Secrets (API Keys)

```bash
# Create OpenAI API Key secret
echo -n "sk-proj-YOUR_OPENAI_KEY_HERE" | gcloud secrets create openai-api-key --data-file=-

# Create Supabase URL secret
echo -n "https://bzyclxmakfklbxnsradh.supabase.co/" | gcloud secrets create supabase-url --data-file=-

# Create Supabase Key secret
echo -n "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." | gcloud secrets create supabase-key --data-file=-

echo "✅ Secrets created"
```

---

### Step 7: Grant Secret Access to Cloud Run

```bash
# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

# Grant access to each secret
for SECRET in openai-api-key supabase-url supabase-key; do
    gcloud secrets add-iam-policy-binding $SECRET \
        --member=serviceAccount:$SERVICE_ACCOUNT \
        --role=roles/secretmanager.secretAccessor
done

echo "✅ Secrets access granted"
```

---

### Step 8: Build Docker Image with Cloud Build

```bash
# Build the image using Google's servers (no local Docker needed!)
# This takes 20-30 minutes first time
gcloud builds submit \
    --tag europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --timeout=45m \
    --machine-type=e2-highcpu-8

echo "✅ Docker image built and pushed"
```

**What's happening:**
- Google's servers build your Docker image
- Installs CUDA, Python, FFmpeg, PyTorch, TalkNet
- Downloads Whisper models
- Pushes to Artifact Registry
- **You just wait!** ☕

---

### Step 9: Deploy to Cloud Run with L4 GPU

```bash
gcloud run deploy clippeak-api \
    --image=europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --region=europe-west1 \
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

echo "✅ Deployed to Cloud Run!"
```

---

### Step 10: Get Your Service URL

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe clippeak-api --region=europe-west1 --format='value(status.url)')

echo "============================================================"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "============================================================"
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test it:"
echo "curl $SERVICE_URL/health"
echo "============================================================"
```

---

## 🧪 Testing Your Deployment

### Test Health Endpoint
```bash
# Should return GPU info
curl https://clippeak-api-XXXXX-ew.a.run.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "gpu_name": "NVIDIA L4",
  "supabase_connected": true,
  "openai_configured": true
}
```

---

## 💰 Storage & Cost Questions Answered

### Where Temporary Files Are Stored:

**Cloud Shell (5GB free):**
- ✅ Only stores your code (~100MB)
- ✅ NOT used for video processing
- ✅ No extra cost

**Container Ephemeral Storage:**
- ✅ Each container gets isolated filesystem
- ✅ Up to memory limit (16GB in our config)
- ✅ **Included in Cloud Run pricing** (no separate charge!)
- ✅ Automatically deleted when container shuts down

### Multiple Users Simultaneously:

**User A uploads → Container 1 starts:**
```
Container 1 filesystem:
├─ tmp/
│  ├─ user_a_video.mp4
│  └─ output_cropped000.mp4
├─ results/
└─ save/pyavi/ (User A's TalkNet files)
```

**User B uploads at same time → Container 2 starts:**
```
Container 2 filesystem (COMPLETELY SEPARATE!):
├─ tmp/
│  ├─ user_b_video.mp4
│  └─ output_cropped000.mp4
├─ results/
└─ save/pyavi/ (User B's TalkNet files)
```

**They NEVER conflict!** Each container = isolated environment.

---

## 🔄 Updating Your Deployment

When you push changes to GitHub:

```bash
# In Cloud Shell
cd ClipPeak
git pull origin main

# Rebuild and redeploy (faster due to caching)
gcloud builds submit \
    --tag europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --timeout=45m

# Deploy update
gcloud run deploy clippeak-api \
    --image=europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --region=europe-west1
```

---

## 📊 Monitoring

### View Logs (Real-time):
```bash
gcloud run services logs tail clippeak-api --region=europe-west1
```

### View Metrics:
Go to: Cloud Run → clippeak-api → Metrics tab

You'll see:
- Request count
- Container instances (should be 0 when idle!)
- GPU utilization
- Memory usage
- Processing time

### View Costs:
Go to: Billing → Reports
- Filter by "Cloud Run"
- Should be $0 when no videos processing!

---

## 🎯 Expected Performance

**With L4 GPU:**
- Cold start: 30-60 seconds (first request after idle)
- Processing: 2-5 minutes per video
- Warm start: Instant (if another video comes within 15 minutes)

**Cost (only when running!):**
- ~$1.20/hour while processing
- Example: 5-minute video = $0.10
- 100 videos/month × 5min = ~$10/month

---

## ⚠️ Troubleshooting

### If build fails:
```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# View specific build
gcloud builds log BUILD_ID
```

### If deployment fails:
```bash
# Check service logs
gcloud run services logs read clippeak-api --region=europe-west1 --limit=50

# Describe service
gcloud run services describe clippeak-api --region=europe-west1
```

### If GPU quota exceeded:
```bash
# Request quota increase
# Go to: console.cloud.google.com → IAM & Admin → Quotas
# Filter: "L4 GPUs"
# Select → "Edit Quotas" → Request increase to 4
```

---

## 🔐 Security Best Practices

1. **Remove hardcoded keys from code** (already done!)
2. **Use Cloud Run secrets** (already configured!)
3. **Enable authentication** (optional):
   ```bash
   # Deploy without --allow-unauthenticated
   # Then use service account for website access
   ```

---

## 📱 Update Your Website

In your frontend code, update the API URL:

```javascript
// config.js or environment variable
const API_URL = process.env.REACT_APP_API_URL || 
    'https://clippeak-api-XXXXX-ew.a.run.app';

// Use it
fetch(`${API_URL}/process-video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ /* your payload */ })
});
```

---

## ✅ Final Checklist

- [ ] Open Cloud Shell
- [ ] Set PROJECT_ID
- [ ] Enable APIs
- [ ] Clone repository
- [ ] Create Artifact Registry
- [ ] Create secrets (API keys)
- [ ] Grant secret access
- [ ] Build with Cloud Build
- [ ] Deploy to Cloud Run
- [ ] Test /health endpoint
- [ ] Update website API URL
- [ ] Test video upload
- [ ] Monitor costs

---

## 🎉 You're Done!

Your system is now:
- ✅ Running on GPU in the cloud
- ✅ Scales to zero when idle (no cost!)
- ✅ Handles multiple users
- ✅ Automatically cleans up
- ✅ Accessible from your website

Total setup time: **~45-60 minutes** (mostly waiting for builds)

