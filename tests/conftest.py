import pytest
from config import Config


@pytest.fixture(scope="module")
def config_load(): 
    config = Config.load_config_class('./tests/test_config.yaml')
