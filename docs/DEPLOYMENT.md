# 🚀 Deployment Guide

## Option 1: Render (Recommended - Easiest)

### Steps:
1. **Push to GitHub**
   ```bash
   cd /Users/juigupte/Desktop/Learning/recsys-foundations
   git init
   git add .
   git commit -m "Initial commit"
   gh repo create music-api --public --source=. --remote=origin --push
   ```

2. **Deploy on Render**
   - Go to https://render.com
   - Sign up (free)
   - Click "New" → "Web Service"
   - Connect your GitHub repo
   - Settings:
     - **Name**: music-identification-api
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

3. **Your API will be live at**: `https://music-identification-api.onrender.com`

---

## Option 2: Railway

1. **Push to GitHub** (same as above)

2. **Deploy on Railway**
   - Go to https://railway.app
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repo
   - Railway auto-detects FastAPI
   - Your API will be live!

---

## Option 3: Fly.io

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and Deploy**
   ```bash
   cd /Users/juigupte/Desktop/Learning/recsys-foundations
   fly auth login
   fly launch
   fly deploy
   ```

---

## 🔧 Important Notes

### Database Files
Your deployment needs these files:
- ✅ `songs.db` (2.30 MB) - Your 24 songs
- ✅ `recommendations.db` (0.26 MB) - 1,000 songs
- ✅ `track_classifier.keras` - CNN model

⚠️ **Warning**: Free tiers have storage limits:
- Render: 512 MB
- Railway: 1 GB
- Fly.io: 3 GB

### Current Project Size
```bash
# Check your project size
du -sh .
```

If too large, consider:
1. Store databases on cloud storage (AWS S3, Google Cloud Storage)
2. Use PostgreSQL instead of SQLite
3. Compress/optimize model files

---

## 🧪 Test Your Deployed API

Once deployed, test with:

```bash
# Replace with your deployment URL
curl https://your-app.onrender.com/health

# Test identification
curl -X POST https://your-app.onrender.com/identify \
  -F "file=@song.mp3"
```

---

## 💡 Pro Tips

1. **Environment Variables**: Set in platform dashboard
2. **CORS**: Already configured in `api.py`
3. **HTTPS**: All platforms provide free SSL
4. **Monitoring**: Check logs in platform dashboard
5. **Auto-deploy**: Platforms auto-deploy on git push

---

## 📱 Next: Build a Frontend

Once deployed, build a web/mobile app:
- React/Next.js web app
- Flutter/React Native mobile app
- Chrome extension

Your API URL: `https://your-app.onrender.com`
