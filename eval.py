import data
from inc.config_snake.config import ConfigFile
from model import Model
import preprocessing as pre
import setup


def interface():
    config_file = ConfigFile("config.json")
    config = config_file.segmentation

    assert config.training.validation_mode == "IID"
    assert config.training.eval_mode == "hung"

    evaluate(config)


def evaluate(config):
    # SETUP
    components = setup.setup(config)
    image_info = components["image_info"]
    heads_info = components["heads_info"]
    output_files = components["output_files"]
    state_folder = components["state_folder"]
    image_folder = components["image_folder"]
    net = components["net"]
    model = Model.load(state_folder, net=net, use_best_net=True)
    preprocessing = pre.SimplePreprocessing(
        image_info=image_info,
        prescale_all=config.preprocessor.prescale_all,
        prescale_factor=config.preprocessor.prescale_factor,
    )
    preprocessor = pre.EvalImagePreprocessor(
        image_info=image_info,
        preprocessing=preprocessing,
        output_files=output_files,
        do_render=config.output.rendering.enabled,
        render_limit=config.output.rendering.limit,
    )
    eval_dataset = data.EvalDataset(
        eval_folder=image_folder,
        preprocessor=preprocessor,
        input_size=heads_info.input_size,
        extensions=config.dataset.extensions,
    )
    eval_dataloader = data.EvalDataLoader(dataset=eval_dataset)
    model.evaluate(output_files=output_files, loader=eval_dataloader)


if __name__ == "__main__":
    interface()
