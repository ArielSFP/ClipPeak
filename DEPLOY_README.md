# 🚀 ClipPeak - Google Cloud Run Deployment

## Quick Start (5 Minutes to Deploy!)

### Option 1: Interactive Setup (Recommended)
```bash
# In Google Cloud Shell (console.cloud.google.com → click terminal icon)
git clone https://github.com/ArielSFP/ClipPeak.git
cd ClipPeak
chmod +x cloud_shell_setup.sh
./cloud_shell_setup.sh
```

The script will guide you through everything! Just answer the prompts.

---

### Option 2: Manual Setup

If you prefer to run commands yourself:

```bash
# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Create registry
gcloud artifacts repositories create clippeak-repo --repository-format=docker --location=europe-west1

# Create secrets (will prompt for values)
echo -n "YOUR_OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=-
echo -n "YOUR_SUPABASE_URL" | gcloud secrets create supabase-url --data-file=-
echo -n "YOUR_SUPABASE_KEY" | gcloud secrets create supabase-key --data-file=-

# Grant access
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
for SECRET in openai-api-key supabase-url supabase-key; do
    gcloud secrets add-iam-policy-binding $SECRET \
        --member=serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
        --role=roles/secretmanager.secretAccessor
done

# Build (takes 20-30 min)
gcloud builds submit --tag europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest --timeout=45m

# Deploy (takes 5-10 min)
gcloud run deploy clippeak-api \
    --image=europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --region=europe-west1 \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --memory=16Gi \
    --cpu=8 \
    --timeout=3600 \
    --max-instances=2 \
    --min-instances=0 \
    --concurrency=1 \
    --set-secrets=OPENAI_API_KEY=openai-api-key:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest \
    --allow-unauthenticated

# Get URL
gcloud run services describe clippeak-api --region=europe-west1 --format='value(status.url)'
```

---

## 📁 Files Created

- ✅ `Dockerfile` - Container configuration
- ✅ `.dockerignore` - Files to exclude from image
- ✅ `deploy.sh` - Simple deployment script
- ✅ `cloud_shell_setup.sh` - Interactive setup wizard
- ✅ `CLOUD_RUN_SETUP.md` - Detailed documentation
- ✅ `QUICK_REFERENCE.md` - Command cheat sheet

---

## 🎯 How It Works

### Architecture
```
User uploads video → clippeak.co.il
                        ↓
                   POST /process-video
                        ↓
              Cloud Run container starts
                        ↓
              Downloads video from Supabase
                        ↓
              Processes with GPU (TalkNet + Whisper)
                        ↓
              Uploads results to Supabase
                        ↓
              Deletes temp files
                        ↓
              Container shuts down → $0 cost!
```

### Storage During Processing

**Each container gets isolated storage:**
- Container has 16GB ephemeral storage
- Includes: tmp/, results/, save/ folders
- Auto-deleted when container stops
- **No separate storage charges!**

**Multiple users = multiple containers:**
- User A → Container 1 → Isolated filesystem
- User B → Container 2 → Isolated filesystem
- No conflicts, no shared files!

---

## 💰 Cost Breakdown

### What You Pay For:

**Artifact Registry** (Docker image storage):
- Your image size: ~8-10GB
- Cost: ~$0.10/GB/month = **$0.80-1.00/month**

**Cloud Run Compute** (only while processing):
- L4 GPU + 16GB RAM + 8 CPU
- Cost: ~$1.20/hour
- Charged **per second** (not full hour!)
- Example: 5-min video = $0.10

**Cloud Build** (rebuilding image):
- First 120 build-minutes/day: FREE
- After that: $0.003/build-minute
- Your builds: ~25 min each = **FREE** (unless rebuilding many times/day)

### What You DON'T Pay For:
- ✅ Idle time (min-instances=0)
- ✅ Container storage (included)
- ✅ Egress to Supabase (same region)
- ✅ Cloud Shell (5GB free)

### Monthly Cost Estimates:
- **Light** (50 videos × 5min): ~$8-10/month
- **Medium** (200 videos × 5min): ~$30-40/month
- **Heavy** (1000 videos × 5min): ~$150-200/month

---

## 🧪 Testing

### Test Health Endpoint
```bash
curl https://YOUR-SERVICE-URL/health
```

Should return:
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

### Test from Your Website
Update your website's API URL to the Cloud Run URL and try uploading a video!

---

## 🔧 Configuration Options

### Current Settings (in deploy.sh):
```bash
--gpu=1                  # Number of GPUs (1 L4)
--gpu-type=nvidia-l4    # GPU type
--memory=16Gi           # RAM (16GB)
--cpu=8                 # CPU cores
--timeout=3600          # Max 60 minutes per request
--max-instances=2       # Max 2 videos simultaneously
--min-instances=0       # Scale to zero when idle
--concurrency=1         # 1 video per container
```

### Adjust for Your Needs:
- More simultaneous users? → Increase `--max-instances`
- Longer videos? → Keep `--timeout=3600` (max allowed)
- Faster processing? → Increase `--memory` and `--cpu`
- Reduce cold starts? → Set `--min-instances=1` (costs more!)

---

## 📊 Monitoring Dashboard

Go to Cloud Console → Cloud Run → clippeak-api

You'll see:
- **Metrics**: Request count, latency, instances
- **Logs**: Real-time processing logs
- **Revisions**: Deployment history
- **YAML**: Full configuration

---

## 🎓 What You Learned

1. ✅ Cloud Run scales to zero (no idle cost!)
2. ✅ Each container = isolated environment
3. ✅ Ephemeral storage is included (free!)
4. ✅ Secrets keep API keys secure
5. ✅ Cloud Build builds without local Docker
6. ✅ L4 GPUs are powerful and cost-effective

---

## 🚨 Important Notes

1. **GPU Quota**: You need L4 quota approved in europe-west1
2. **Cold Start**: First request takes 30-60s (container startup + model loading)
3. **Max Timeout**: 60 minutes (Cloud Run limit)
4. **Container Size**: ~8-10GB (due to CUDA + models)
5. **Build Time**: First build: 25-30 min, subsequent: 10-15 min (caching)

---

## ✅ Ready to Deploy?

1. Open Cloud Shell
2. Run `./cloud_shell_setup.sh`
3. Wait for build & deploy (~45 minutes)
4. Get your service URL
5. Update your website
6. Test it!
7. Monitor costs

**Questions?** Check logs, read CLOUD_RUN_SETUP.md, or view Cloud Run documentation.

Good luck! 🎉

