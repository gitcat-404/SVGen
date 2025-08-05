from openai import AsyncOpenAI, AsyncAzureOpenAI

def get_client(config):
    provider = config["provider"]

    if provider == "openai":
        return AsyncOpenAI(api_key=config["openai"]["api_key"])

    elif provider == "azure":
        azure_cfg = config["azure"]
        return AsyncAzureOpenAI(
            api_key=azure_cfg["api_key"],
            api_version=azure_cfg["api_version"],
            base_url=f"{azure_cfg['api_base']}/openai/deployments/{azure_cfg['deployment_name']}"
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
