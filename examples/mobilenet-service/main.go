package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/juandes/tensorflow-go-models/models"
)

// After starting the service, you can test it with curl:
// curl -F "data=@static/images/person.jpg" http://localhost:8080/predict
var model *models.Coco

const port = 8080

func main() {
	model = models.NewCoco()
	err := model.Load()
	if err != nil {
		fmt.Printf("Error loading model: %v", err)
		panic(err)
	}

	defer model.CloseSession()
	router := mux.NewRouter()
	router.
		Path("/predict").
		Methods("POST").
		HandlerFunc(predict)

	fmt.Printf("Listening on port %d\n", port)
	http.ListenAndServe(fmt.Sprintf(":%s", strconv.Itoa(port)), router)
}

func predict(w http.ResponseWriter, r *http.Request) {
	defer elapsed()()
	file, _, err := r.FormFile("data")
	if err != nil {
		http.Error(w, "Unable to get file", http.StatusInternalServerError)
		return
	}

	fileBytes, err := ioutil.ReadAll(file)
	if err != nil {
		http.Error(w, "Unable to read file", http.StatusInternalServerError)
		return
	}

	outcome := model.Predict(fileBytes)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(outcome)
}

func elapsed() func() {
	start := time.Now()
	return func() {
		fmt.Printf("Elapsed time %v\n", time.Since(start))
	}
}
