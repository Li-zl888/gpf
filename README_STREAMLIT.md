# Streamlit Deployment Guide for Guangdong Population Flow Analysis

This guide provides instructions for deploying the Guangdong Population Flow Analysis application on Streamlit Cloud.

## Prerequisites

- A GitHub account
- The project code pushed to a GitHub repository
- A Streamlit Cloud account (sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud))

## Files for Streamlit Deployment

The following files are required for Streamlit deployment:

1. `app_streamlit.py` - The main Streamlit application file
2. `requirements_streamlit.txt` - The requirements file for Streamlit Cloud
3. `.streamlit/config.toml` - Configuration settings for Streamlit

## Deployment Steps

1. **Push your code to GitHub**

   Make sure your project is pushed to a GitHub repository, including the Streamlit app file (`app_streamlit.py`), requirements file (`requirements_streamlit.txt`), and Streamlit configuration file (`.streamlit/config.toml`).

2. **Sign in to Streamlit Cloud**

   Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. **Deploy your app**

   - Click on "New app"
   - Select your GitHub repository
   - Select the branch (usually `main` or `master`)
   - Enter the path to your Streamlit app file (`app_streamlit.py`)
   - Click "Deploy"

4. **Configure your app (optional)**

   - Set environment variables if needed
   - Adjust advanced settings like Python version, memory limits, etc.

5. **Access your deployed app**

   Once deployed, your app will be available at a URL like `https://username-app-name-streamlit-app.streamlit.app`.

## Customizing Your Deployment

### Requirements File

The `requirements_streamlit.txt` file specifies the Python packages required for your app. Make sure it includes all necessary dependencies:

```
streamlit==1.22.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0
plotly==5.3.1
trafilatura==1.0.0
beautifulsoup4==4.10.0
requests==2.26.0
```

### Streamlit Configuration

The `.streamlit/config.toml` file allows you to customize the appearance and behavior of your Streamlit app:

```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Troubleshooting

- **Missing dependencies**: Make sure all required packages are listed in `requirements_streamlit.txt`
- **Import errors**: Check that all imported modules are available on Streamlit Cloud
- **File not found errors**: Ensure all file paths are relative and files are included in your repository
- **Memory issues**: Reduce memory usage or request more resources in Streamlit Cloud settings

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community](https://discuss.streamlit.io/)