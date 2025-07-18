package main

import (
	"context"
	"fmt"
	"log"
	"os"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

// WeatherRequest represents the parameters for the getWeather function
type WeatherRequest struct {
	Location string `json:"location"`
}

func main() {
	dotenv.Load()
	// Initialize Gemini client with API key from environment
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY environment variable is required")
	}

	swarm := swarmgo.NewSwarm(apiKey, llm.Gemini)

	// Example 1: Basic chat completion
	fmt.Println("Example 1: Basic Chat Completion")
	agent := &swarmgo.Agent{
		Name:         "Agent",
		Instructions: "You are a helpful agent.",
		Model:        "gemini-pro",
	}

	messages := []llm.Message{
		{Role: llm.RoleUser, Content: "Hi!"},
	}

	ctx := context.Background()
	response, err := swarm.Run(ctx, agent, messages, nil, "", false, false, 5, true)
	if err != nil {
		panic(err)
	}

	fmt.Println(response.Messages[len(response.Messages)-1].Content)

	/* 	// Example 2: Function calling
	   	fmt.Println("\nExample 2: Function Calling")

	   	weatherAgent := &swarmgo.Agent{
	   		Name:         "WeatherAgent",
	   		Instructions: "You are a weather assistant. When asked about weather, always use the getWeather function.",
	   		Model:        "gemini-pro",
	   		Functions: []swarmgo.AgentFunction{
	   			{
	   				Name:        "getWeather",
	   				Description: "Get the current weather for a location",
	   				Parameters: map[string]interface{}{
	   					"type": "object",
	   					"properties": map[string]interface{}{
	   						"location": map[string]interface{}{
	   							"type":        "string",
	   							"description": "The city and state/country",
	   						},
	   					},
	   					"required": []interface{}{"location"},
	   				},
	   				Function: func(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	   					location := args["location"].(string)
	   					return swarmgo.Result{
	   						Success: true,
	   						Data:    fmt.Sprintf(`{"location": "%s", "temperature": "65"}`, location),
	   					}
	   				},
	   			},
	   		},
	   	}

	   	swarmgo.RunDemoLoop(swarm, weatherAgent) */

	// Example 3: Streaming completion
	fmt.Println("\nExample 3: Streaming Completion")

	streamAgent := &swarmgo.Agent{
		Name:         "StreamAgent",
		Instructions: "You are a helpful agent that responds in a streaming fashion.",
		Model:        "gemini-pro",
	}

	streamMessages := []llm.Message{
		{Role: llm.RoleUser, Content: "Count from 1 to 5 slowly"},
	}

	streamResponse, err := swarm.Run(ctx, streamAgent, streamMessages, nil, "", true, false, 5, true)
	if err != nil {
		panic(err)
	}

	fmt.Println(streamResponse.Messages[len(streamResponse.Messages)-1].Content)

}
