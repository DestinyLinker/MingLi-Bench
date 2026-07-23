import os
import unittest
from unittest.mock import patch

from mingli_bench.models.atlascloud_client import AtlasCloudClient
from mingli_bench.models.factory import ModelFactory
from mingli_bench.utils.config import load_config


ATLAS_ENV_KEYS = [
    "ATLASCLOUD_API_KEY",
    "ATLAS_CLOUD_API_KEY",
    "ATLASCLOUD_API_BASE",
    "ATLASCLOUD_BASE_URL",
    "ATLAS_CLOUD_API_BASE",
    "ATLAS_CLOUD_BASE_URL",
]


class TestAtlasCloudProvider(unittest.TestCase):
    def setUp(self):
        self.original_registry_entry = ModelFactory._registry["atlascloud"]

    def tearDown(self):
        ModelFactory._registry["atlascloud"] = self.original_registry_entry

    def test_load_config_reads_atlascloud_aliases(self):
        env = {
            key: ""
            for key in ATLAS_ENV_KEYS
        }
        env.update(
            {
                "ATLAS_CLOUD_API_KEY": "atlas-key",
                "ATLASCLOUD_BASE_URL": "https://atlas.example/v1",
                "MAX_TOKENS": "1024",
                "TEMPERATURE": "0.2",
            }
        )

        with patch.dict(os.environ, env, clear=False):
            config = load_config(env_file="/tmp/mingli-bench-no-env-file")

        self.assertEqual(config["atlascloud"]["api_key"], "atlas-key")
        self.assertEqual(config["atlascloud"]["base_url"], "https://atlas.example/v1")
        self.assertEqual(config["atlascloud"]["max_tokens"], 1024)
        self.assertEqual(config["atlascloud"]["temperature"], 0.2)

    def test_factory_detects_atlascloud_prefixes(self):
        self.assertEqual(ModelFactory.get_provider("atlascloud/qwen/qwen3.5-flash"), "atlascloud")
        self.assertEqual(ModelFactory.get_provider("atlas-cloud/deepseek-ai/deepseek-v4-pro"), "atlascloud")
        self.assertEqual(ModelFactory.get_provider("atlas/qwen/qwen3.5-flash"), "atlascloud")

    def test_factory_strips_atlascloud_routing_prefix(self):
        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model_name = kwargs["model_name"]

        ModelFactory._registry["atlascloud"] = FakeClient

        client = ModelFactory.create(
            "atlascloud/qwen/qwen3.5-flash",
            config={
                "atlascloud": {
                    "api_key": "atlas-key",
                    "base_url": "https://api.atlascloud.ai/v1",
                }
            },
        )

        self.assertEqual(client.model_name, "qwen/qwen3.5-flash")
        self.assertEqual(client.kwargs["api_key"], "atlas-key")
        self.assertEqual(client.kwargs["base_url"], "https://api.atlascloud.ai/v1")

    def test_atlascloud_client_initializes_openai_compatible_client(self):
        with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "atlas-key"}, clear=False):
            with patch("mingli_bench.models.atlascloud_client.OpenAI") as openai_cls:
                client = AtlasCloudClient(model_name="qwen/qwen3.5-flash")

        self.assertEqual(client.api_key, "atlas-key")
        self.assertEqual(client.base_url, "https://api.atlascloud.ai/v1")
        openai_cls.assert_called_once_with(
            api_key="atlas-key",
            base_url="https://api.atlascloud.ai/v1",
            timeout=30,
        )


if __name__ == "__main__":
    unittest.main()
