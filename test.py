from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor
overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": "sam3.pt", "half": True, "save": True}
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image
predictor.set_image("/home/new_users/zhiyu/projects/ultralytics/dog-puppy-on-garden-royalty-free-image-1586966191.avif")

# Provide bounding box examples to segment similar objects
results = predictor(text=["person"])

# # Multiple bounding boxes for different concepts
# results = predictor(bboxes=[[539, 599, 589, 639], [343, 267, 499, 662]])
