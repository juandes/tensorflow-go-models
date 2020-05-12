package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/juandes/tensorflow-go-models/models"
)

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

	fmt.Printf("Listeing on port %d", port)
	http.ListenAndServe(fmt.Sprintf(":%s", strconv.Itoa(port)), router)
}

func predict(w http.ResponseWriter, r *http.Request) {
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
