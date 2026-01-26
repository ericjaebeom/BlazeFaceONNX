from onnxscript import opset22 as op
from onnxscript import script


@script()
def Softplus(X):
    return op.Log(op.Exp(X) + 1.0)

softplus = Softplus.to_function_proto()