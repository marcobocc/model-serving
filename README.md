# Setup

Download and prepare the model for serving:

```shell
./setup/emotions_classifier.sh 
```

Start the model server:

```shell
./start.sh 
```

# Inference
Send a POST request to `127.0.0.1:8080/predictions/emotions_classifier` with JSON body:
```json
{
    "input" : ["Hello world!"]
}
```
