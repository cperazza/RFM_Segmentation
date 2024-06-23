#!/bin/zsh

# variables
model_name="llama3:8B"
custom_model_name="crewai-llama3-8B"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./Llama3ModelFile