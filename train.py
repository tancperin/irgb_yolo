data_yaml_path = "irgb\data.yaml"

from ultralytics import YOLO

def train_yolov11s(
	data_yaml=data_yaml_path,
	model_name="yolo11s.pt",
	epochs=100,  
	imgsz=640,
	batch=4, 
	project="runs/train",
	name="irgb",
	device="cuda", 
	workers=2  
):
	"""
	Trains YOLOv11s using Ultralytics on the specified dataset.
	"""
	model = YOLO(model_name)  
	results = model.train(
		data=data_yaml,
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		project=project,
		name=name,
		device=device,
		workers=workers
	)
	print("Training complete. Results saved to:", results.save_dir)


if __name__ == "__main__":
	train_yolov11s()

