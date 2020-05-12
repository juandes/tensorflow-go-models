package models

import (
	"fmt"
	"path"
	"path/filepath"
	"runtime"

	"github.com/juandes/tensorflow-go-models/responses"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	_, b, _, _ = runtime.Caller(0)
	basepath   = filepath.Dir(b)
)

const modelPath = "../static/models/ssd_mobilenet_v1_coco_2018_01_28"

// Coco is a MobileNet V1 model trained on the COCO dataset.
type Coco struct {
	model  *tf.SavedModel
	labels []string
}

// NewCoco returns a Coco object
func NewCoco() *Coco {
	return &Coco{}
}

// Load loads the ssd_mobilenet_v1_coco_2018_01_28 SavedModel.
func (c *Coco) Load() error {
	fmt.Println(path.Join(basepath, "..", modelPath))
	model, err := tf.LoadSavedModel(path.Join(basepath, modelPath, "/"), []string{"serve"}, nil)
	if err != nil {
		return fmt.Errorf("Error loading model: %v", err)
	}
	c.model = model

	c.labels, err = readLabels(path.Join(basepath, modelPath, "labels.txt"))
	if err != nil {
		return fmt.Errorf("Error loading labels file: %v", err)
	}
	return nil
}

// Predict predicts.
func (c *Coco) Predict(data []byte) *responses.ObjectDetectionResponse {
	tensor, _ := makeTensorFromBytes(data)

	output, err := c.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			c.model.Graph.Operation("image_tensor").Output(0): tensor,
		},
		[]tf.Output{
			c.model.Graph.Operation("detection_boxes").Output(0),
			c.model.Graph.Operation("detection_classes").Output(0),
			c.model.Graph.Operation("detection_scores").Output(0),
			c.model.Graph.Operation("num_detections").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running the session: %v", err)
		return nil
	}

	outcome := responses.NewObjectDetectionResponse(output, c.labels)
	return outcome
}

// CloseSession closes a session.
func (c *Coco) CloseSession() {
	c.model.Session.Close()
}
