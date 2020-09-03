你好！
很冒昧用这样的方式来和你沟通，如有打扰请忽略我的提交哈。我是光年实验室（gnlab.com）的HR，在招Golang开发工程师，我们是一个技术型团队，技术氛围非常好。全职和兼职都可以，不过最好是全职，工作地点杭州。
我们公司是做流量增长的，Golang负责开发SAAS平台的应用，我们做的很多应用是全新的，工作非常有挑战也很有意思，是国内很多大厂的顾问。
如果有兴趣的话加我微信：13515810775  ，也可以访问 https://gnlab.com/，联系客服转发给HR。
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
