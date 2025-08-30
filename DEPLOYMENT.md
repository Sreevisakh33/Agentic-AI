# 🚀 Deployment Guide for Hugging Face Spaces

This guide will help you deploy your Career Chatbot with RAG to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Career Documents**: Ensure you have `me/linkedin.pdf` and `me/summary.txt`

## 🎯 Step-by-Step Deployment

### 1. Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Choose **"Gradio"** as the SDK
4. Set **Space name** to: `career-chatbot-rag`
5. Set **Space SDK** to: `Gradio`
6. Set **License** to: `MIT`
7. Click **"Create Space"**

### 2. Upload Your Files

1. In your new Space, go to the **"Files"** tab
2. Upload these files to the root directory:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Create a `me/` folder and upload:
   - `me/linkedin.pdf`
   - `me/summary.txt`

### 3. Set Environment Variables

1. Go to **"Settings"** tab in your Space
2. Scroll to **"Repository secrets"**
3. Add a new secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key
4. Click **"Add secret"**

### 4. Deploy

1. Go to **"App"** tab
2. Your app should automatically build and deploy
3. Wait for the build to complete (usually 2-5 minutes)
4. Your chatbot will be available at the provided URL

## 🔧 Troubleshooting

### Build Fails
- Check that all required files are uploaded
- Verify `requirements.txt` has correct dependencies
- Check the build logs for specific errors

### App Won't Start
- Ensure `OPENAI_API_KEY` is set correctly
- Check that career documents exist in `me/` folder
- Review the app logs for runtime errors

### Missing Dependencies
- Verify `requirements.txt` includes all necessary packages
- Check that Gradio version is compatible (4.19.2)

## 🌟 Features After Deployment

✅ **Automatic Document Loading**: LinkedIn PDF and summary text  
✅ **Document Upload**: Add more career documents  
✅ **AI-Powered Responses**: Personalized career advice  
✅ **Smart Context Retrieval**: Relevant information finding  
✅ **Modern UI**: Beautiful, responsive interface  
✅ **Authentication**: Password protection  

## 🔐 Default Login

- **Password**: `career2024`
- You can change this in the `app.py` file

## 📱 Access Your App

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/career-chatbot-rag
```

## 🆘 Need Help?

- Check the build logs in your Space
- Review the README.md for feature details
- Ensure all files are properly uploaded
- Verify environment variables are set correctly

---

**Happy Deploying! 🎉**
