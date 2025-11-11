# ClipPeak Cloud Run - Quick Reference

## 🚀 Initial Setup (One Time Only)

```bash
# In Google Cloud Shell
cd ~
git clone https://github.com/ArielSFP/ClipPeak.git
cd ClipPeak
chmod +x cloud_shell_setup.sh
./cloud_shell_setup.sh
```

Follow the prompts. Total time: ~45-60 minutes.

---

## 🔄 Update After Code Changes

```bash
# In Cloud Shell
cd ~/ClipPeak
git pull origin main

# Quick redeploy (takes ~15-20 minutes)
export PROJECT_ID="your-project-id"
gcloud builds submit \
    --tag europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --timeout=45m

gcloud run deploy clippeak-api \
    --image=europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest \
    --region=europe-west1
```

---

## 📊 Monitoring Commands

### View Real-Time Logs
```bash
gcloud run services logs tail clippeak-api --region=europe-west1
```

### Check Service Status
```bash
gcloud run services describe clippeak-api --region=europe-west1
```

### Get Service URL
```bash
gcloud run services describe clippeak-api --region=europe-west1 --format='value(status.url)'
```

### Test Health Endpoint
```bash
SERVICE_URL=$(gcloud run services describe clippeak-api --region=europe-west1 --format='value(status.url)')
curl $SERVICE_URL/health
```

---

## 💰 Cost Monitoring

### View Current Month Costs
```bash
# Go to console.cloud.google.com/billing/reports
# Filter by "Cloud Run" service
```

### Check Running Instances
```bash
# Should be 0 when idle!
gcloud run services describe clippeak-api --region=europe-west1 --format='value(status.traffic[0].latestRevision)'
```

---

## 🛠️ Useful Operations

### Update Secrets (API Keys)
```bash
# Update OpenAI key
echo -n "NEW_KEY_HERE" | gcloud secrets versions add openai-api-key --data-file=-

# Update Supabase key
echo -n "NEW_KEY_HERE" | gcloud secrets versions add supabase-key --data-file=-

# Redeploy to use new secrets
gcloud run services update clippeak-api --region=europe-west1
```

### Scale Up (More Instances)
```bash
# Allow up to 4 concurrent videos
gcloud run services update clippeak-api \
    --region=europe-west1 \
    --max-instances=4
```

### Scale Down (Reduce Max)
```bash
# Back to 2
gcloud run services update clippeak-api \
    --region=europe-west1 \
    --max-instances=2
```

### Increase Timeout (for longer videos)
```bash
# Max is 3600 seconds (60 minutes)
gcloud run services update clippeak-api \
    --region=europe-west1 \
    --timeout=3600
```

---

## 🐛 Debugging

### View Recent Errors
```bash
gcloud run services logs read clippeak-api \
    --region=europe-west1 \
    --limit=100 \
    --format=json | grep -i error
```

### Test Locally in Cloud Shell (before deploying)
```bash
# Build locally
docker build -t test-clippeak .

# Run locally (no GPU, but tests basic functionality)
docker run -p 8080:8080 -e PORT=8080 test-clippeak

# Test in Cloud Shell preview
# Click "Web Preview" → "Preview on Port 8080"
```

---

## 📈 Performance Tuning

### Current Settings (Optimized for Cost)
- Memory: 16Gi
- CPU: 8
- Min instances: 0 (scale to zero)
- Max instances: 2

### If Processing is Slow
```bash
# Increase resources
gcloud run services update clippeak-api \
    --region=europe-west1 \
    --memory=32Gi \
    --cpu=16
```

### If You Want Faster Cold Starts
```bash
# Keep 1 instance always warm (costs ~$30/day)
gcloud run services update clippeak-api \
    --region=europe-west1 \
    --min-instances=1
```

---

## 🔥 Emergency: Stop Everything

### Delete Service (stop all costs)
```bash
gcloud run services delete clippeak-api --region=europe-west1
```

### Delete Image (free up storage)
```bash
gcloud artifacts docker images delete \
    europe-west1-docker.pkg.dev/$PROJECT_ID/clippeak-repo/clippeak-api:latest
```

---

## 📞 Getting Help

### Check Cloud Build History
```bash
gcloud builds list --limit=10
```

### View Specific Build Log
```bash
BUILD_ID="abc-123-456"
gcloud builds log $BUILD_ID
```

### Check GPU Quota
```bash
gcloud compute project-info describe --project=$PROJECT_ID
```

---

## 🎯 Common Issues & Solutions

### "Quota exceeded for L4 GPUs"
**Solution:** Request quota increase at console.cloud.google.com/iam-admin/quotas

### "Container failed to start"
**Solution:** Check logs with `gcloud run services logs read clippeak-api --region=europe-west1 --limit=50`

### "Build timeout"
**Solution:** Increase timeout with `--timeout=60m` in build command

### "Out of memory"
**Solution:** Increase memory with `--memory=32Gi` in deploy command

---

## ✅ Health Check Examples

### Good Response:
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

### Bad Response (No GPU):
```json
{
  "status": "healthy",
  "gpu_available": false,
  "gpu_count": 0,
  "gpu_name": "None"
}
```
**Fix:** Check GPU quota and deployment configuration.

---

## 🎉 That's It!

Your Cloud Run service is now:
- ⚡ GPU-accelerated
- 💰 Cost-optimized (scale to zero)
- 🌍 Globally accessible
- 📊 Fully monitored
- 🔐 Securely configured

**Support:** If issues persist, check logs and Cloud Run documentation.

