import eval
import train
from inc.config_snake.config import ConfigFile

# TODO automate testing


def run():
    config_file = ConfigFile("./test/test_config.json")
    config = config_file.segmentation

    assert config.training.validation_mode == "IID"
    assert config.training.eval_mode == "hung"

    print("testing training")
    train.train(config)

    print("testing eval")
    eval.evaluate(config)


if __name__ == "__main__":
    run()
