# config.py

CONFIG = {
    "provider": "azure",  # "openai" 或 "azure"

    "openai": {
        "api_key": "your-openai-key"
    },

    "azure": {
        "api_base": "https://xxx.openai.azure.com",
        "api_key": "your-azure-key",
        "deployment_name": "gpt-4o",
        "api_version": "2024-02-15-preview"
    }
}
