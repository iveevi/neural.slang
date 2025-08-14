import trimesh
import argparse
from common import *


def main(Network, RenderingPipeline, TrainingPipeline):
    device = create_device()
    
    network = Network(
        device,
        hidden=32,
        hidden_layers=2,
        levels=8,
        input=3,
        output=3,
    )
    
    rendering_pipeline = RenderingPipeline(device, network)
    training_pipeline = TrainingPipeline(device, network)
    
    assert device.has_feature(spy.Feature.acceleration_structure)
    
    geometry = trimesh.load_mesh(ROOT / "resources" / "spinosa.obj")
    print("geometry", geometry)
    print("geometry.visual", geometry.visual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=[
        "addresses",
    ], default="addresses")
    args = parser.parse_args()

    match args.network:
        case "addresses":
            from .addresses import Network, RenderingPipeline, TrainingPipeline
            main(Network, RenderingPipeline, TrainingPipeline)
        case _:
            raise ValueError(f"Invalid network: {args.network}")