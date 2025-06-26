BASE_URL = https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip
MODELS = mobileclip_s0.pt mobileclip_s1.pt mobileclip_s2.pt mobileclip_b.pt mobileclip_blt.pt

ONNX_MODELS = $(MODELS:.pt=.onnx)
ONNX_TARGETS = $(addprefix onnx/,$(ONNX_MODELS))

QUANTIZED_ONNX_MODELS = $(ONNX_MODELS:.onnx=_int8.onnx)
QUANTIZED_ONNX_TARGETS = $(addprefix onnx/,$(QUANTIZED_ONNX_MODELS))

.PHONY: install download onnx quantize

venv/bin/activate:
	python3 -m venv venv

install: venv/bin/activate
	source venv/bin/activate && \
	pip install -r requirements.txt

download: $(addprefix tensorflow/,$(MODELS))

tensorflow/%.pt:
	curl -L $(BASE_URL)/$(@F) -o $@ --create-dirs

onnx: $(ONNX_TARGETS)

onnx/%.onnx: tensorflow/%.pt venv/bin/activate
	mkdir -p onnx
	source venv/bin/activate && \
	python export_to_onnx.py --checkpoint $< --onnx $@

quantize: $(QUANTIZED_ONNX_TARGETS)

onnx/%_int8.onnx: onnx/%.onnx venv/bin/activate
	source venv/bin/activate && \
	python quantize_onnx.py --input $< --output $@
