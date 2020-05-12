package responses

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// ObjectDetectionResponse is the response the user receives after requesting an
// object detection prediction
type ObjectDetectionResponse struct {
	Detections    []detection `json:"detections"`
	NumDetections int         `json:"numDetections"`
}

type detection struct {
	Score float32   `json:"score"`
	Box   []float32 `json:"box"`
	Label string    `json:"label"`
}

const threshold = 0.50

// [yMin, xMin, yMax, xMax]

// NewObjectDetectionResponse creates an ObjectDetectionResponse
func NewObjectDetectionResponse(output []*tf.Tensor, labels []string) *ObjectDetectionResponse {
	detectionsAboveThreshold := 0

	detections := []detection{}

	// Use type assertion to get the values of the output tensor.
	outputDetectionBoxes := output[0].Value().([][][]float32)
	outputDetectionClasses := output[1].Value().([][]float32)
	outputDetectionScores := output[2].Value().([][]float32)
	numDetections := int(output[3].Value().([]float32)[0])

	for i := 0; i < numDetections; i++ {
		if outputDetectionScores[0][i] < threshold {
			continue
		}

		detectionsAboveThreshold++

		detection := detection{
			Score: outputDetectionScores[0][i],
			Box:   outputDetectionBoxes[0][i],
			Label: labels[int(outputDetectionClasses[0][i]-1)],
		}
		detections = append(detections, detection)
	}

	return &ObjectDetectionResponse{
		Detections:    detections,
		NumDetections: detectionsAboveThreshold,
	}
}
