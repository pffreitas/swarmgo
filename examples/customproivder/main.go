package main

import (
	"context"
	"fmt"
	"os"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

func main() {
	dotenv.Load()

	aiProvider := llm.NewOpenAILLM(os.Getenv("OPENAI_API_KEY"))

	client := swarmgo.NewSwarmWithCustomProvider(aiProvider, &swarmgo.Config{})

	agent := &swarmgo.Agent{
		Name:         "Agent",
		Instructions: "You are a helpful agent.",
		Model:        "gpt-3.5-turbo",
	}

	messages := []llm.Message{
		{Role: llm.RoleUser, Content: "Hi!"},
	}

	ctx := context.Background()
	response, err := client.Run(ctx, agent, messages, nil, "", false, false, 5, true)
	if err != nil {
		panic(err)
	}

	fmt.Println(response.Messages[len(response.Messages)-1].Content)
}
