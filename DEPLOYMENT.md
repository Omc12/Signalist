# 🚀 Deploying Signalist to Vercel

## Frontend Deployment (Vercel)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add Vercel configuration"
git push origin main
```

### Step 2: Deploy to Vercel

#### Option A: Vercel CLI (Fastest)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from frontend folder
cd frontend
vercel

# For production
vercel --prod
```

#### Option B: Vercel Dashboard
1. Go to https://vercel.com
2. Click "Add New Project"
3. Import your GitHub repository `Omc12/Signalist`
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Add Environment Variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: Your backend URL (see below)

### Step 3: Set Backend URL

After deploying your backend (Render/Railway), add the URL to Vercel:
```bash
vercel env add VITE_API_BASE_URL
# Enter: https://your-backend.onrender.com
```

Or in Vercel Dashboard:
- Go to Project Settings → Environment Variables
- Add `VITE_API_BASE_URL` = `https://your-backend-url.com`

---

## Backend Options for Vercel Frontend

### Option 1: Render (Recommended - Free Tier)
```bash
# 1. Create render.yaml in project root (see below)
# 2. Push to GitHub
# 3. Connect to Render.com
# 4. Copy the backend URL
# 5. Add to Vercel env: VITE_API_BASE_URL
```

### Option 2: Railway.app
```bash
# 1. Connect Railway to GitHub
# 2. Deploy backend folder
# 3. Copy provided URL
# 4. Add to Vercel env
```

### Option 3: Vercel Serverless (Advanced)
Convert FastAPI to serverless functions (requires restructuring)

---

## Important: CORS Configuration

Update your backend CORS to allow Vercel domain:

```python
# backend/core/config.py
CORS_ORIGINS = [
    "http://localhost:5173",
    "https://your-app.vercel.app",  # Add your Vercel URL
    "https://*.vercel.app"          # Allow all Vercel preview deployments
]
```

---

## Testing Your Deployment

1. **Frontend**: `https://your-app.vercel.app`
2. **Backend**: `https://your-backend.onrender.com/health`
3. **API Connection**: Check browser console for errors

---

## Common Issues

### CORS Errors
- Add your Vercel URL to backend CORS_ORIGINS
- Redeploy backend after changes

### API Not Found
- Check VITE_API_BASE_URL is set correctly
- Ensure backend is deployed and running

### Build Fails
- Check Node.js version (use 18+)
- Verify package.json scripts

---

## Next Steps

1. ✅ Frontend configured for Vercel
2. 📤 Deploy backend (Render/Railway)
3. 🔗 Update VITE_API_BASE_URL with backend URL
4. 🚀 Deploy to Vercel
5. 🎉 Your app is live!
