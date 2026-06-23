from export.cann_export import CannModelSpec, input_shape_argument


def test_input_shape_argument_supports_arbitrary_model_inputs() -> None:
    spec = CannModelSpec(
        input_names=("image", "conditioning"),
        output_names=("detections",),
        input_shapes={"image": (1, 3, 640, 640), "conditioning": (1, 16)},
    )

    assert input_shape_argument(spec) == "image:1,3,640,640;conditioning:1,16"
