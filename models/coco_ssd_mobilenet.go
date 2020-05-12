package models

import (
	"fmt"
	"strings"

	"github.com/juandes/tensorflow-models/responses"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Coco is a MobileNet V1 model trained on the COCO dataset.
type Coco struct {
	model  *tf.SavedModel
	labels []string
}

const path = "static/models/ssd_mobilenet_v1_coco_2018_01_28/"

// NewCoco returns a Coco object
func NewCoco() *Coco {
	return &Coco{}
}

// Load loads the ssd_mobilenet_v1_coco_2018_01_28 SavedModel.
func (c *Coco) Load() error {
	model, err := tf.LoadSavedModel(path, []string{"serve"}, nil)
	if err != nil {
		return fmt.Errorf("Error loading model: %v", err)
	}
	c.model = model
	c.labels, err = readLabels(strings.Join([]string{path, "labels.txt"}, ""))
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
