package models

import (
	"fmt"
	"io/ioutil"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Functions taken from here https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/example_inception_inference_test.go

// makeBatch uses ExpandDims to convert the tensor into a batch of size 1.
func makeBatch() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

// Convert the image in filename to a Tensor suitable as input
func makeTensorFromBytes(bytes []byte) (*tf.Tensor, error) {
	// bytes to tensor
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}

	// create batch
	graph, input, output, err := makeBatch()
	if err != nil {
		return nil, err
	}

	// Execute that graph create the batch of that image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	defer session.Close()

	batch, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return batch[0], nil
}

func readLabels(labelsFile string) ([]string, error) {
	fileBytes, err := ioutil.ReadFile(labelsFile)
	if err != nil {
		return nil, fmt.Errorf("Unable to read labels file: %v", err)
	}

	return strings.Split(string(fileBytes), "\n"), nil
}
