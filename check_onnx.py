import onnx
m = onnx.load("/Users/condenast/Downloads/policy1.onnx")
with open("/Users/condenast/Downloads/onnx_info.txt", "w") as f:
    f.write("=== policy1.onnx metadata ===\n")
    for e in m.metadata_props:
        f.write(e.key + " = " + repr(e.value[:300]) + "\n")
    f.write("\n")
    for inp in m.graph.input:
        s = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        f.write("INPUT " + inp.name + " " + str(s) + "\n")
    for out in m.graph.output:
        s = [d.dim_value for d in out.type.tensor_type.shape.dim]
        f.write("OUTPUT " + out.name + " " + str(s) + "\n")
print("done -> /Users/condenast/Downloads/onnx_info.txt")
