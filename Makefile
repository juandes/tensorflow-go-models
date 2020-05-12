start-server:
	go run examples/mobilenet-service/main.go

tags:
	saved_model_cli show --dir static/models/ssd_mobilenet_v1_coco_2018_01_28 --all 

testC:
	gcc hello_tf.c -ltensorflow -o hello_tf
	./hello_tf