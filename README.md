# TensorFlow models for TensorFlow Go

## Overview
The purpose of this project is to host several TensorFlow models that can be loaded
out of the box to use with TensorFlow Go.

## Models
- MobileNet object detection trained on COCO.

## To install

To install the library use `$ go get github.com/juandes/tensorflow-go-models/models`

## Usage
You can load and use a model like this:

```go
package main

import (
	"fmt"

	"github.com/juandes/tensorflow-go-models/models"
)

var model *models.Coco

func main() {
	model = models.NewCoco()
	err := model.Load()
	if err != nil {
		fmt.Printf("Error loading model: %v", err)
		panic(err)
	}

	defer model.CloseSession()
}

```


In the `cmd/` directory you can find an example of a web service that serves the MobileNet model.
For more information about how to use it check out the blog post [Using TensorFlow Go to serve an object detection model with a web service](https://juandes.com/tensorflow-go-object-detection-server/)