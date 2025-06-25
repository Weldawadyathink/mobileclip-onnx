BASE_URL = https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip
MODELS = mobileclip_s0.pt mobileclip_s1.pt mobileclip_s2.pt mobileclip_b.pt mobileclip_blt.pt

.PHONY: install download

venv/bin/activate:
	python3 -m venv venv

install: venv/bin/activate
	source venv/bin/activate && \
	pip install -r requirements.txt

download: $(addprefix tensorflow/,$(MODELS))

tensorflow/%.pt:
	curl -L $(BASE_URL)/$(@F) -o $@ --create-dirs
