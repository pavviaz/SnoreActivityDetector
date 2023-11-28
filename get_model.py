import torch

from ml_pipeline.model_manager import ModelManager


manager = ModelManager(model_name="M5E_greek_clear")

model = manager.get_featured_model()
model.to("cpu")
model.eval()

dummy_inp = torch.rand(*manager.get_model_inp_shape())

traced_model = torch.jit.trace(model, dummy_inp)
traced_model.save("M5E_greek_clear_jit.pt")

torch.onnx.export(
    model,  # model being run
    dummy_inp,  # model input (or a tuple for multiple inputs)
    "M5E_greek_clear.onnx",  # where to save the model
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=False,  # whether to execute constant folding for optimization
    # input_names=["modelInput"],  # the model's input names
    # output_names=["modelOutput"],  # the model's output names
    # dynamic_axes={
    #     "modelInput": {0: "batch_size"},  # variable length axes
    #     "modelOutput": {0: "batch_size"},
    # },
)

print()
