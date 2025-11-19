# VibeMatch Deployment Guide

## 🚀 Recommended: Render (Backend) + Vercel (Frontend)

This is the easiest and most reliable approach.

---

## Step 1: Deploy Backend API to Render

### Prerequisites
- GitHub account
- Render account (free): https://render.com

### Steps

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign up/Login with GitHub

2. **Create Web Service**
   - Click "New +" → "Web Service"
   - Click "Connect a repository"
   - Select: `gitjui/VibeMatch`

3. **Configure Service**
   - **Name**: `vibematch-api`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

4. **Advanced Settings** (scroll down)
   - Add environment variable:
     - Key: `PYTHON_VERSION`
     - Value: `3.11.0`

5. **Create Web Service**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build
   - Note your API URL: `https://vibematch-api.onrender.com`

---

## Step 2: Deploy Frontend to Vercel

### Prerequisites
- Vercel account (free): https://vercel.com

### Steps

1. **Go to Vercel Dashboard**
   - Visit https://vercel.com/new
   - Sign up/Login with GitHub

2. **Import Repository**
   - Click "Import Project"
   - Select: `gitjui/VibeMatch`
   - Click "Import"

3. **Configure Project**
   - **Project Name**: `vibematch`
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

4. **Environment Variables**
   - Click "Environment Variables"
   - Add variable:
     - Name: `VITE_API_URL`
     - Value: `https://vibematch-api.onrender.com` (your Render API URL)

5. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes
   - Your app is live: `https://vibematch.vercel.app`

---

## Alternative: Render Only (Both Services)

### Deploy Backend (API)

1. **Create Web Service**
   - New → Web Service
   - Connect repo: `gitjui/VibeMatch`
   - Name: `vibematch-api`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**
   - `PYTHON_VERSION`: `3.13.3`

3. **Deploy** - Click "Create Web Service"

### Deploy Frontend

1. **Create Web Service**
   - New → Web Service
   - Connect repo: `gitjui/VibeMatch`
   - Name: `vibematch-frontend`
   - Root Directory: `frontend`
   - Runtime: Node
   - Build Command: `npm install && npm run build`
   - Start Command: `npm run preview`

2. **Environment Variables**
   - `VITE_API_URL`: `https://vibematch-api.onrender.com` (your API URL)

3. **Deploy** - Click "Create Web Service"

---

## 📦 What Gets Deployed

### Backend Files
- ✅ FastAPI application (`api.py`)
- ✅ Python dependencies (`requirements.txt`)
- ✅ Database files (`songs.db`, `recommendations.db`)
- ✅ CNN model (`track_classifier.keras`)
- ✅ All Python modules

### Frontend Files
- ✅ React app (Vite build)
- ✅ Tailwind CSS
- ✅ All components and assets

---

## ⚠️ Important Notes

### Free Tier Limits (Render)
- **Spins down after 15 minutes of inactivity**
- First request after sleep takes 30-60 seconds
- 750 hours/month free
- 512 MB RAM
- Upgrade to paid plan ($7/month) for always-on service

### Database Files
Current size:
- `songs.db`: ~2.3 MB (24 songs with embeddings)
- `recommendations.db`: ~260 KB (1,000 songs)
- `track_classifier.keras`: ~1.2 MB
- **Total**: ~3.8 MB ✅ (well within limits)

---

## 🧪 Testing Your Deployment

Once deployed, test:

```bash
# Test API health
curl https://vibematch-api.onrender.com/health

# Test frontend
# Visit: https://vibematch-frontend.onrender.com

# Test API directly
curl -X POST https://vibematch-api.onrender.com/identify \
  -F "file=@your-song.mp3"
```

---

## 🔧 Troubleshooting

### Issue: CORS errors
- Make sure API CORS allows your frontend domain
- Check `api.py` CORS configuration

### Issue: API not found (404)
- Verify API URL in frontend environment variables
- Check `VITE_API_URL` is set correctly

### Issue: Slow first load
- Normal for free tier - app sleeps after inactivity
- Upgrade to paid plan for always-on service

---

## 💰 Cost

**Free Option**:
- Both services free on Render
- Limitations: sleep after inactivity, slower

**Paid Option** ($7-14/month):
- Always-on services
- Faster response times
- More resources

---

## 🎯 Next Steps

After deployment:
1. Share your app URL with friends
2. Test with different songs
3. Monitor usage in Render dashboard
4. Consider upgrading if needed

Enjoy your deployed VibeMatch! 🎵
